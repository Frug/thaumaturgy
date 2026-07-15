"""llama.cpp server management + chat streaming.

We reuse textgen's approach: spawn the bundled `llama-server` binary as a
subprocess and talk to it over HTTP. Generation goes through llama-server's
OpenAI-compatible `/v1/chat/completions` endpoint, so the model's own chat
template and sampling are handled by llama.cpp — we just pass messages + params.

Single model at a time (one subprocess); matches local single-user use.
"""

import atexit
from collections import deque
import json
import os
import signal
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

import llama_cpp_binaries

from thaumaturgy import appstate, metadata_gguf
from thaumaturgy.paths import sub_dir

SERVER_LOG_LIMIT = 500


def models_dir():
    return sub_dir("models")


def _pidfile() -> Path:
    return sub_dir("cache") / "llama_server.pid"


def _reap_stale() -> None:
    """Kill a llama-server orphaned by a previous app instance.

    Hot reload recreates the LlamaServer singleton with an empty handle while
    the old subprocess keeps running (and holding VRAM). We record each server's
    PID in a file; on startup we terminate a leftover one — but only if the PID
    is still an actual llama-server, to guard against PID reuse.
    """
    pf = _pidfile()
    try:
        pid = int(pf.read_text())
    except (OSError, ValueError):
        return
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
        if "llama-server" in cmd:
            os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    finally:
        pf.unlink(missing_ok=True)


def list_models() -> list[str]:
    return sorted(p.name for p in models_dir().glob("*.gguf"))


_ctx_cache: dict[str, int | None] = {}
_max_gpu_layers_cache: dict[str, int | None] = {}


def trained_ctx(model_name: str) -> int | None:
    """Model's trained context length, read from GGUF metadata (cached)."""
    if model_name not in _ctx_cache:
        try:
            _ctx_cache[model_name] = metadata_gguf.read_context_length(
                models_dir() / model_name)
        except OSError:
            _ctx_cache[model_name] = None
    return _ctx_cache[model_name]


def max_gpu_layers(model_name: str) -> int | None:
    """Maximum GPU layers for llama.cpp, based on GGUF block count."""
    if model_name not in _max_gpu_layers_cache:
        try:
            blocks = metadata_gguf.read_block_count(models_dir() / model_name)
            _max_gpu_layers_cache[model_name] = blocks + 1 if blocks is not None else None
        except OSError:
            _max_gpu_layers_cache[model_name] = None
    return _max_gpu_layers_cache[model_name]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class LlamaServer:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.model: str | None = None
        self.n_ctx: int | None = None  # trained/effective context, learned after load
        self._log_lines: deque[str] = deque(maxlen=SERVER_LOG_LIMIT)
        self._log_lock = threading.Lock()
        self._log_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def output_lines(self) -> list[str]:
        with self._log_lock:
            return list(self._log_lines)

    def _clear_output(self) -> None:
        with self._log_lock:
            self._log_lines.clear()

    def _append_output(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text:
            return
        with self._log_lock:
            self._log_lines.append(text)

    def _capture_output(self, stream) -> None:
        try:
            for line in stream:
                self._append_output(line)
        except ValueError:
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def start(self, model_name: str, gpu_layers: int = -1,
              ctx_size: int = 0, cache_type: str = "fp16") -> None:
        self.stop()
        path = models_dir() / model_name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        port = _free_port()
        cmd = [
            llama_cpp_binaries.get_binary_path(),
            "-m", str(path),
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(gpu_layers),
            "-c", str(ctx_size),
        ]
        if cache_type and cache_type != "fp16":
            cmd += ["-ctk", cache_type, "-ctv", cache_type]
        self._clear_output()
        self._append_output("$ " + " ".join(shlex.quote(str(part)) for part in cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     bufsize=1)
        if self.proc.stdout is not None:
            self._log_thread = threading.Thread(target=self._capture_output,
                                                args=(self.proc.stdout,),
                                                daemon=True)
            self._log_thread.start()
        self.port = port
        self.model = model_name
        _pidfile().write_text(str(self.proc.pid))
        self._wait_ready()
        self._read_props()
        appstate.state.current_model = model_name

    def _wait_ready(self, timeout: float = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running:
                raise RuntimeError("llama-server exited during startup")
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become ready in time")

    def _read_props(self) -> None:
        try:
            props = requests.get(f"{self.base_url}/props", timeout=10).json()
            self.n_ctx = (props.get("default_generation_settings", {}).get("n_ctx")
                          or props.get("n_ctx"))
        except (requests.RequestException, ValueError):
            self.n_ctx = None

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            self._append_output("Stopping llama-server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._append_output("llama-server did not stop in time; killing it.")
                proc.kill()
                proc.wait(timeout=5)
            self._append_output("llama-server stopped.")
        elif proc is not None:
            self._append_output(f"llama-server exited with code {proc.returncode}.")
        self.proc = None
        self.port = None
        self.model = None
        self.n_ctx = None
        _pidfile().unlink(missing_ok=True)

    def stream_chat(self, messages: list[dict], params: dict | None = None):
        """Yield streaming chat events from /v1/chat/completions (SSE)."""
        p = params or {}
        body = {
            "messages": messages,
            "stream": True,
            "temperature": p.get("temperature", 0.8),
            "top_p": p.get("top_p", 0.95),
            "top_k": int(p.get("top_k", 40)),
            "min_p": p.get("min_p", 0.05),
            "repeat_penalty": p.get("repetition_penalty", 1.1),
            "max_tokens": int(p.get("max_new_tokens", 512)),
        }
        finish_reason = None
        with requests.post(f"{self.base_url}/v1/chat/completions",
                           json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    yield {"type": "delta", "text": delta}
        if finish_reason:
            yield {"type": "finish", "reason": finish_reason}


_reap_stale()  # clean up a llama-server orphaned by a previous (reloaded) instance
server = LlamaServer()
atexit.register(server.stop)
