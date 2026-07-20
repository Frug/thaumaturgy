"""llama.cpp server management + chat streaming.

We reuse textgen's approach: spawn a resolved `llama-server` binary as a
subprocess and talk to it over HTTP. Generation goes through llama-server's
OpenAI-compatible `/v1/chat/completions` endpoint, so the model's own chat
template and sampling are handled by llama.cpp — we just pass messages + params.

Single model at a time (one subprocess); matches local single-user use.
"""

import atexit
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import requests

from thaumaturgy import appstate, llama_bins, metadata_gguf, store
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


# A split model's parts, e.g. "model-Q4_K_M-00001-of-00003.gguf".
SHARD_RE = re.compile(r"(?i)-\d{5}-of-\d{5}\.gguf$")


def list_models() -> list[str]:
    """Loadable models, one entry per model rather than per file.

    llama.cpp opens a split model through its first part and finds the rest
    itself, so listing every part would offer parts 2..n as models that cannot
    load. Sorting puts the lowest part first; if part 1 is missing the set is
    broken anyway, and listing what's there keeps it visible and deletable.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in sorted(p.name for p in models_dir().glob("*.gguf")):
        if not SHARD_RE.search(name):
            out.append(name)
            continue
        stem = SHARD_RE.sub("", name)
        if stem not in seen:
            seen.add(stem)
            out.append(name)
    return out


def model_files(name: str) -> list[Path]:
    """Every file backing one model — a whole shard set, or a lone file.

    Removing just the named part of a split model would strand the other parts
    as unloadable orphans, so deletion has to work on the set.
    """
    if not SHARD_RE.search(name):
        path = models_dir() / name
        return [path] if path.exists() else []
    stem = SHARD_RE.sub("", name)
    return sorted(p for p in models_dir().glob("*.gguf")
                  if SHARD_RE.search(p.name) and SHARD_RE.sub("", p.name) == stem)


def delete_model(name: str) -> list[str]:
    """Delete a model's file(s) from disk; returns the names removed.

    Refuses while llama-server holds it open: unlink would succeed but the
    space would stay claimed until the server exits, so the disk wouldn't
    actually free and the model would keep serving from a deleted inode.
    """
    files = model_files(name)
    if not files:
        raise RuntimeError(f"{name} is already gone.")
    if server.running and server.model:
        if (models_dir() / server.model) in files:
            raise RuntimeError(f"{server.model} is loaded — unload it first.")
    removed = []
    for path in files:
        path.unlink(missing_ok=True)
        removed.append(path.name)
        _drop_cached(path.name)
    return removed


# Keyed by (name, mtime_ns, size) so a re-downloaded or still-copying file is
# re-read rather than serving the previous file's metadata.
_ctx_cache: dict[tuple, int | None] = {}
_max_gpu_layers_cache: dict[tuple, int | None] = {}


def _drop_cached(name: str) -> None:
    for cache in (_ctx_cache, _max_gpu_layers_cache):
        for key in [k for k in cache if k[0] == name]:
            del cache[key]


def _read_metadata(cache: dict, model_name: str, read) -> int | None:
    """Cache a GGUF metadata read, keyed on the file's identity.

    Failures are not cached: a partially-copied download or a briefly
    unavailable mount would otherwise pin the answer to None for the life of
    the process. The parser raises struct.error / KeyError / ValueError on a
    malformed or truncated file, and the callers render into the page on every
    refresh, so nothing may escape here.
    """
    path = models_dir() / model_name
    try:
        stat = path.stat()
        key = (model_name, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    if key not in cache:
        try:
            cache[key] = read(path)
        except Exception:  # noqa: BLE001 - any malformed GGUF, never a broken page
            return None
    return cache[key]


def trained_ctx(model_name: str) -> int | None:
    """Model's trained context length, read from GGUF metadata (cached)."""
    return _read_metadata(_ctx_cache, model_name, metadata_gguf.read_context_length)


def max_gpu_layers(model_name: str) -> int | None:
    """Maximum GPU layers for llama.cpp, based on GGUF block count."""
    blocks = _read_metadata(_max_gpu_layers_cache, model_name,
                            metadata_gguf.read_block_count)
    return blocks + 1 if blocks is not None else None


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
        self.chat_template_caps: dict = {}
        self.reasoning_budget: int = -1  # thinking cap the server was launched with
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
              ctx_size: int = 0, cache_type: str = "fp16",
              chat_template: str = "auto", reasoning: str = "auto",
              reasoning_budget: int = -1,
              reasoning_budget_message: str = "") -> None:
        self.stop()
        path = models_dir() / model_name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        port = _free_port()
        cmd = [
            str(llama_bins.server_path()),
            "-m", str(path),
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(gpu_layers),
            "-c", str(ctx_size),
        ]
        if cache_type and cache_type != "fp16":
            cmd += ["-ctk", cache_type, "-ctv", cache_type]
        if chat_template and chat_template != "auto":
            cmd += ["--chat-template", chat_template]
        if reasoning and reasoning != "auto":
            cmd += ["--reasoning", reasoning]
        if reasoning_budget != -1:
            cmd += ["--reasoning-budget", str(reasoning_budget)]
        # Without this the forced end-of-thinking tag lands mid-sentence; the
        # message gives the model a cue to wrap up. Pointless with no budget.
        if reasoning_budget > 0 and reasoning_budget_message:
            cmd += ["--reasoning-budget-message", reasoning_budget_message]
        self.reasoning_budget = reasoning_budget
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
        store.save_last_loaded_model(model_name)

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
            caps = props.get("chat_template_caps") or {}
            self.chat_template_caps = caps if isinstance(caps, dict) else {}
        except (requests.RequestException, ValueError):
            self.n_ctx = None
            self.chat_template_caps = {}

    def supports_system_role(self) -> bool:
        """Whether the active llama.cpp chat template accepts a system message."""
        if not self.running:
            return True
        supported = self.chat_template_caps.get("supports_system_role")
        return True if supported is None else bool(supported)

    @staticmethod
    def _wait(proc: subprocess.Popen, timeout: float) -> bool:
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _shut_down(self, proc: subprocess.Popen) -> None:
        """Terminate proc, escalating to SIGKILL, and report the outcome."""
        self._append_output("Stopping llama-server...")
        proc.terminate()
        if not self._wait(proc, 10):
            self._append_output("llama-server did not stop in time; killing it.")
            proc.kill()
            if not self._wait(proc, 5):
                # Only reachable if the process is wedged in an uninterruptible
                # wait. Give up rather than block the UI (or exit) forever.
                self._append_output("llama-server ignored SIGKILL; abandoning it.")
                return
        self._append_output("llama-server stopped.")

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            self._shut_down(proc)
        elif proc is not None:
            self._append_output(f"llama-server exited with code {proc.returncode}.")
        self.proc = None
        self.port = None
        self.model = None
        self.n_ctx = None
        self.chat_template_caps = {}
        self.reasoning_budget = -1
        _pidfile().unlink(missing_ok=True)

    @staticmethod
    def _fallback_chat_prompt(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        parts.append("assistant:\n")
        return "\n\n".join(parts)

    @staticmethod
    def _token_count_from_json(data) -> int | None:
        if isinstance(data, list):
            return len(data)
        if not isinstance(data, dict):
            return None
        for key in ("n_tokens", "num_tokens", "token_count", "count"):
            value = data.get(key)
            if isinstance(value, int):
                return value
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        if isinstance(tokens, int):
            return tokens
        return None

    @staticmethod
    def _text_from_json(data) -> str | None:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return None
        for key in ("prompt", "content", "text"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        return None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def _error_message(response) -> str:
        """Pull llama-server's rejection reason out of an error response body.

        The status code alone rarely says why the request was refused.
        """
        try:
            body = response.json()
        except ValueError:
            return response.text.strip()[:500] or "no detail"
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            error = error.get("message") or error
        return str(error if error else body)[:500]

    def count_chat_tokens(self, messages: list[dict]) -> tuple[int, bool]:
        """Return prompt-token usage for a chat, with exactness flag.

        llama-server can apply the active chat template and tokenize the result.
        If either endpoint is unavailable, fall back to a conservative text
        estimate so the UI still has a useful context meter before load.
        """
        prompt = self._fallback_chat_prompt(messages)
        if self.running:
            try:
                r = requests.post(
                    f"{self.base_url}/apply-template",
                    json={"messages": messages, "add_generation_prompt": True},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                count = self._token_count_from_json(data)
                if count is not None:
                    return count, True
                prompt = self._text_from_json(data) or prompt
            except (requests.RequestException, ValueError, TypeError):
                pass

            for payload in (
                {"content": prompt, "add_special": False},
                {"content": prompt},
            ):
                try:
                    r = requests.post(f"{self.base_url}/tokenize", json=payload, timeout=10)
                    r.raise_for_status()
                    count = self._token_count_from_json(r.json())
                    if count is not None:
                        return count, True
                except (requests.RequestException, ValueError, TypeError):
                    pass

        return self._estimate_tokens(prompt), False

    def _max_tokens(self, max_new_tokens: int) -> int:
        """Grow the request cap to cover thinking as well as the reply.

        max_tokens caps thinking and reply together, so a bare max_new_tokens
        lets a long thought eat the whole allowance and leave nothing to answer
        with. A thinking model needs the sum. With no budget set (-1) there is
        no bound to add, and a long enough thought can still exhaust the cap.
        """
        return max_new_tokens + max(0, self.reasoning_budget)

    def stream_chat(self, messages: list[dict], params: dict | None = None):
        """Yield streaming chat events from /v1/chat/completions (SSE).

        Emits "reasoning" events only for templates llama.cpp can build a parser
        for; others (Gemma's) leave channel markers in the reply for the caller
        to split.
        """
        p = params or {}
        body = {
            "messages": messages,
            "stream": True,
            "temperature": p.get("temperature", 0.8),
            "top_p": p.get("top_p", 0.95),
            "top_k": int(p.get("top_k", 40)),
            "min_p": p.get("min_p", 0.05),
            "repeat_penalty": p.get("repetition_penalty", 1.1),
            "max_tokens": self._max_tokens(int(p.get("max_new_tokens", 512))),
        }
        finish_reason = None
        with requests.post(f"{self.base_url}/v1/chat/completions",
                           json=body, stream=True, timeout=600) as r:
            if not r.ok:
                raise RuntimeError(
                    f"llama-server returned {r.status_code}: {self._error_message(r)}")
            for raw_line in r.iter_lines(decode_unicode=False):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")
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
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                content = delta.get("content")
                if content:
                    yield {"type": "delta", "text": content}
        if finish_reason:
            yield {"type": "finish", "reason": finish_reason}


_reap_stale()  # clean up a llama-server orphaned by a previous (reloaded) instance
server = LlamaServer()
atexit.register(server.stop)
