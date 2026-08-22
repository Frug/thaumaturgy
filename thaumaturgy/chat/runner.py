"""One assistant reply: its thread, its accumulating text, its persistence."""

import threading
import time
from collections.abc import Callable

from thaumaturgy import engine
from thaumaturgy.chat import reply
from thaumaturgy.chat.models import TOKEN_CAP_LIMIT, Message

SAVE_INTERVAL = 0.5  # partial output survives a crash mid-reply


class ChatRun:
    """A streaming reply, written into its Message as it arrives.

    The page reads the Message, so it sees the reply grow without needing to
    know anything about the stream.
    """

    def __init__(self, chat_id: str, message: Message, index: int,
                 api_messages: list[dict], params: dict,
                 on_persist: Callable[[], None] | None = None,
                 on_done: Callable[[], None] | None = None):
        self.chat_id = chat_id
        self.message = message
        self.index = index
        self.api_messages = api_messages
        self.params = params
        self.on_persist = on_persist
        self.on_done = on_done
        self.done = False
        self.error: str | None = None
        self._raw_text = message.text or ""
        self._raw_reasoning = message.reasoning or ""
        self._thread: threading.Thread | None = None

    def start(self) -> "ChatRun":
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        return self

    @property
    def snapshot(self) -> tuple[str, str]:
        """What the page should currently show for this reply."""
        return self.message.text, self.message.reasoning

    def _render(self, *, streaming: bool) -> None:
        text, reasoning = reply.interpret(self._raw_text, self._raw_reasoning,
                                          streaming=streaming)
        self.message.text = text
        self.message.reasoning = reasoning

    def _persist(self) -> None:
        if self.on_persist is not None:
            self.on_persist()

    def _work(self) -> None:
        last_save = 0.0
        try:
            for event in engine.server.stream_chat(self.api_messages, self.params):
                kind = event.get("type")
                if kind == "finish":
                    self.message.finish_reason = event.get("reason")
                    self.message.finish_limit = event.get("limit", TOKEN_CAP_LIMIT)
                    continue
                delta = event.get("text", "")
                if not delta:
                    continue
                if kind == "reasoning":
                    self._raw_reasoning += delta
                else:
                    self._raw_text += delta
                self._render(streaming=True)
                now = time.monotonic()
                if now - last_save > SAVE_INTERVAL:
                    self._persist()
                    last_save = now
        except Exception as exc:  # noqa: BLE001 - surfaced to the observing page
            self.error = str(exc)
            self.message.finish_reason = "error"
            self.message.generation_error = str(exc)
        finally:
            self._render(streaming=False)
            self._persist()
            self.done = True
            if self.on_done is not None:
                self.on_done()
