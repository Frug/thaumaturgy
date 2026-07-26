"""Turning a chat and its scenario into chat-completion messages."""

from thaumaturgy.chat.models import Chat, Message, Role, Scenario


def _prepend_to_first_user(messages: list[dict], prefix: str) -> list[dict]:
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = f"{prefix}\n\n{msg['content']}".strip()
            return messages
    if prefix:
        messages.insert(0, {"role": "user", "content": prefix})
    return messages


def build(chat: Chat, scenario: Scenario | None, *, draft: str = "",
          supports_system_role: bool = True) -> list[dict]:
    """Assemble the request body's messages, including any unsent `draft`.

    The draft is folded in here rather than appended by the caller: without a
    system role the scenario merges into the first user turn, which may be the
    draft itself.
    """
    system_parts = []
    if scenario is not None and scenario.context.strip():
        system_parts.append(scenario.context.strip())

    history = list(chat.messages)
    # Gemma-style templates raise on a leading assistant turn and no capability
    # flag reports it, so an opening line always moves into the prompt.
    while history and history[0].role is not Role.USER:
        opening = (history.pop(0).text or "").strip()
        if opening:
            system_parts.append(f"Opening scene:\n{opening}")

    messages = []
    for m in history:
        if not (m.text or "").strip() and m.generation_error:
            continue
        messages.append({"role": str(m.role), "content": m.text or ""})
    if draft:
        messages.append({"role": str(Role.USER), "content": draft})

    system = "\n\n".join(system_parts).strip()
    if not system:
        return messages
    if supports_system_role:
        return [{"role": "system", "content": system}, *messages]
    return _prepend_to_first_user(messages, system)


def opening_message(scenario: Scenario | None) -> Message | None:
    if scenario is None or not scenario.opening_text.strip():
        return None
    return Message(role=Role.ASSISTANT, name=scenario.name,
                   text=scenario.opening_text)
