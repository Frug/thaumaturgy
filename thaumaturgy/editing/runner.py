"""One span generation: its thread, its accumulating text, its cancellation."""

import threading
import time

from thaumaturgy import engine


class SpanRun:
    """A single in-flight rewrite.

    The messages are kept after the run ends so the page can show what was
    actually sent rather than rebuilding it from state that has moved on.
    """

    def __init__(self, index: int, messages: list[dict], params: dict):
        self.index = index
        self.messages = messages
        self.params = params
        self.started = time.monotonic()
        self.text = ""
        self.reasoning = ""
        self.finish_reason: str | None = None
        self.done = False
        self.cancelled = False
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started)

    def start(self) -> "SpanRun":
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Stop at the next streamed token."""
        if not self.done:
            self.cancelled = True

    def _work(self) -> None:
        try:
            for event in engine.server.stream_chat(self.messages, self.params):
                if self.cancelled:
                    # Leaving the generator closes the stream and frees the slot.
                    break
                kind = event.get("type")
                if kind == "finish":
                    self.finish_reason = event.get("reason")
                    continue
                delta = event.get("text", "")
                if not delta:
                    continue
                if kind == "reasoning":
                    self.reasoning += delta
                else:
                    self.text += delta
        except Exception as exc:  # noqa: BLE001 - surfaced to the observing page
            self.error = str(exc)
        finally:
            self.done = True
