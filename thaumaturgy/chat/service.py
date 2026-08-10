"""The chat service: owns the open chat, its scenario, and in-flight replies.

Replies are tracked per chat rather than one at a time, so switching away from a
generating chat and back keeps it; the page re-attaches to whatever is running.

One instance for the process, like engine.server.
"""

from enum import StrEnum, auto

from thaumaturgy import appstate, engine, store
from thaumaturgy.chat import prompt
from thaumaturgy.chat.models import Chat, Message, Role, Scenario
from thaumaturgy.chat.runner import ChatRun
from thaumaturgy.lang import en
from thaumaturgy.outcome import Outcome


class Step(StrEnum):
    STARTED = auto()   # a reply is streaming
    BLOCKED = auto()   # no model, or this chat is already generating
    UPDATED = auto()   # state changed, nothing generating
    IDLE = auto()      # nothing open
    ERROR = auto()


class ChatService:
    """Owns the conversation the page is showing, and any reply in flight."""

    def __init__(self):
        self.chat: Chat | None = None
        self.scenario_name: str | None = None
        self._runs: dict[str, ChatRun] = {}

    # ── scenarios ────────────────────────────────────────────────────────────
    def scenarios(self) -> list[Scenario]:
        return [Scenario.from_dict(s) for s in store.list_scenarios()]

    def scenario(self) -> Scenario | None:
        for s in self.scenarios():
            if s.name == self.scenario_name:
                return s
        return None

    def select_scenario(self, name: str | None) -> Outcome:
        self.scenario_name = name
        appstate.state.current_scenario = name
        store.save_last_scenario(name)
        return self.open_first(name)

    # ── chats ────────────────────────────────────────────────────────────────
    def list_chats(self, scenario: str | None = None) -> list[dict]:
        return store.list_chats(scenario)

    def _adopt(self, chat: Chat | None) -> Outcome:
        self.chat = chat
        appstate.state.current_chat_id = chat.id if chat else None
        return Outcome(Step.UPDATED)

    def open(self, chat_id: str | None) -> Outcome:
        """Open a chat, preferring the live copy a running reply writes into."""
        if chat_id is None:
            return self._adopt(None)
        run = self._runs.get(chat_id)
        if run is not None and self.chat is not None and self.chat.id == chat_id:
            return Outcome(Step.UPDATED)
        raw = store.load_chat(chat_id)
        if raw is None:
            return Outcome(Step.ERROR, "That chat could not be loaded.")
        return self._adopt(Chat.from_dict(raw))

    def open_first(self, scenario: str | None) -> Outcome:
        chats = store.list_chats(scenario)
        if not chats:
            return self._adopt(None)
        # Keep the live object if that chat is mid-reply.
        if self.chat is not None and self.chat.id == chats[0]["id"] \
                and chats[0]["id"] in self._runs:
            return Outcome(Step.UPDATED)
        return self._adopt(Chat.from_dict(chats[0]))

    def new_chat(self) -> Outcome:
        scenario = self.scenario()
        raw = store.new_chat(self.scenario_name, appstate.state.current_model, None)
        chat = Chat.from_dict(raw)
        opening = prompt.opening_message(scenario)
        if opening is not None:
            chat.append(opening)
            self._save(chat)
        return self._adopt(chat)

    def delete(self, chat_id: str) -> Outcome:
        if self.run_for(chat_id) is not None:
            return Outcome(Step.BLOCKED,
                           "Wait for generation to finish before deleting this chat.")
        was_open = self.chat is not None and self.chat.id == chat_id
        store.delete_chat(chat_id)
        if was_open:
            return self.open_first(self.scenario_name)
        return Outcome(Step.UPDATED)

    def rename(self, chat_id: str, title: str) -> Outcome:
        # Blocked mid-reply like delete is: the run holds its own Chat object
        # and would write the derived title back over this one.
        if self.run_for(chat_id) is not None:
            return Outcome(Step.BLOCKED,
                           "Wait for generation to finish before renaming this chat.")
        if not store.rename_chat(chat_id, title):
            return Outcome(Step.ERROR, "That chat could not be renamed.")
        if self.chat is not None and self.chat.id == chat_id:
            self.chat.title = title.strip()
            self.chat.title_custom = True
        return Outcome(Step.UPDATED)

    def _save(self, chat: Chat | None = None) -> None:
        chat = chat or self.chat
        if chat is not None:
            raw = chat.to_dict()
            store.save_chat(raw)
            chat.title = raw.get("title", chat.title)
            chat.updated = raw.get("updated", chat.updated)

    # ── generation ───────────────────────────────────────────────────────────
    def run_for(self, chat_id: str | None) -> ChatRun | None:
        return self._runs.get(chat_id or "")

    @property
    def run(self) -> ChatRun | None:
        return self._runs.get(self.chat.id) if self.chat else None

    def busy(self, chat_id: str | None = None) -> bool:
        run = self._runs.get(chat_id or (self.chat.id if self.chat else ""))
        return run is not None and not run.done

    def _finish(self, chat_id: str) -> None:
        self._runs.pop(chat_id, None)
        if chat_id in appstate.state.generations:
            del appstate.state.generations[chat_id]

    def _reply(self) -> Outcome:
        """Append an empty assistant message and stream into it."""
        chat = self.chat
        scenario = self.scenario()
        api = prompt.build(chat, scenario,
                           supports_system_role=engine.server.supports_system_role())
        message = chat.append(Message(
            role=Role.ASSISTANT, name=self.scenario_name or "",
            model=engine.server.model or appstate.state.current_model))
        self._save()
        run = ChatRun(chat.id, message, len(chat.messages) - 1, api,
                      dict(appstate.state.current_params),
                      on_persist=lambda: self._save(chat))
        self._runs[chat.id] = run
        # Shared with the editing service, which refuses to start while the one
        # llama-server is busy.
        appstate.state.generations[chat.id] = run
        run.start()
        return Outcome(Step.STARTED)

    def send(self, text: str) -> Outcome:
        if not engine.server.running:
            return Outcome(Step.BLOCKED, en.NO_MODEL)
        if not text.strip():
            return Outcome(Step.IDLE)
        if self.chat is None:
            self.new_chat()
        if self.busy(self.chat.id):
            # A second worker on the same chat would interleave its writes with
            # the first's and evict it from the registry.
            return Outcome(Step.BLOCKED, en.CHAT_BUSY)
        self.chat.append(Message(role=Role.USER, name="You", text=text))
        self._save()
        return self._reply()

    def regenerate(self) -> Outcome:
        if self.chat is None:
            return Outcome(Step.IDLE)
        if not engine.server.running:
            return Outcome(Step.BLOCKED, en.NO_MODEL)
        if self.busy(self.chat.id):
            return Outcome(Step.BLOCKED, en.CHAT_BUSY)
        index = self.chat.latest_assistant_index()
        if index is None:
            return Outcome(Step.BLOCKED,
                           "Only the latest assistant reply can be regenerated.")
        self.chat.messages.pop(index)
        self._save()
        return self._reply()

    def complete_run(self, run: ChatRun) -> Outcome:
        """Retire a finished reply. The message already holds its own text."""
        self._finish(run.chat_id)
        self._save()
        if run.error:
            return Outcome(Step.ERROR, f"Generation error: {run.error}")
        return Outcome(Step.UPDATED)

    # ── editing an existing reply ────────────────────────────────────────────
    def edit_last(self, text: str) -> Outcome:
        if self.chat is None:
            return Outcome(Step.IDLE)
        if self.busy(self.chat.id):
            return Outcome(Step.BLOCKED, en.CHAT_BUSY)
        index = self.chat.latest_assistant_index()
        if index is None:
            return Outcome(Step.BLOCKED,
                           "Only the latest assistant reply can be edited.")
        if not text.strip():
            return Outcome(Step.BLOCKED, "Response text can't be empty.")
        message = self.chat.messages[index]
        message.text = text
        message.clear_generation_state()
        self._save()
        return Outcome(Step.UPDATED)

    def edit_message(self, index: int, text: str) -> Outcome:
        """Rewrite one of the user's own messages, wherever it sits."""
        if self.chat is None:
            return Outcome(Step.IDLE)
        if self.busy(self.chat.id):
            return Outcome(Step.BLOCKED, en.CHAT_BUSY)
        if not 0 <= index < len(self.chat.messages):
            return Outcome(Step.ERROR, "That message is no longer there.")
        message = self.chat.messages[index]
        if message.role is not Role.USER:
            return Outcome(Step.BLOCKED, "Only your own messages can be edited here.")
        if not text.strip():
            return Outcome(Step.BLOCKED, "Message text can't be empty.")
        message.text = text
        self._save()
        return Outcome(Step.UPDATED)

    def editable_reply(self) -> Message | None:
        if self.chat is None:
            return None
        index = self.chat.latest_assistant_index()
        return self.chat.messages[index] if index is not None else None

    # ── context meter ────────────────────────────────────────────────────────
    def context_messages(self, draft: str = "") -> list[dict]:
        chat = self.chat or Chat(id="")
        return prompt.build(chat, self.scenario(), draft=draft,
                            supports_system_role=engine.server.supports_system_role())

    def context_total(self) -> int | None:
        if engine.server.n_ctx:
            return engine.server.n_ctx
        model = engine.server.model or appstate.state.current_model
        return engine.trained_ctx(model) if model else None


chat = ChatService()
