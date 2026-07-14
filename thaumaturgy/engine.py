"""llama.cpp server management + chat streaming.

We reuse textgen's approach: spawn the bundled `llama-server` binary as a
subprocess and talk to it over HTTP. Generation goes through llama-server's
OpenAI-compatible `/v1/chat/completions` endpoint, so the model's own chat
template and sampling are handled by llama.cpp — we just pass messages + params.

Single model at a time (one subprocess); matches local single-user use.
"""

import atexit
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

import llama_cpp_binaries

from thaumaturgy import appstate, metadata_gguf
from thaumaturgy.paths import sub_dir


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

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

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
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.port = None
        self.model = None
        self.n_ctx = None
        _pidfile().unlink(missing_ok=True)

    def stream_chat(self, messages: list[dict], params: dict | None = None):
        """Yield content-delta strings from /v1/chat/completions (SSE)."""
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
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta


_reap_stale()  # clean up a llama-server orphaned by a previous (reloaded) instance
server = LlamaServer()
atexit.register(server.stop)
