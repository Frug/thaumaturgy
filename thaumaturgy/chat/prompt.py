"""Turning a chat and its scenario into chat-completion messages."""

from thaumaturgy import prompting
from thaumaturgy.chat.models import Chat, Message, Role, Scenario


def build(chat: Chat, scenario: Scenario | None, *, draft: str = "",
          supports_system_role: bool = True, compacted: bool = True) -> list[dict]:
    """Assemble the request body's messages, including any unsent `draft`.

    The draft is folded in here rather than appended by the caller: without a
    system role the scenario merges into the first user turn, which may be the
    draft itself.

    With `compacted`, an active recap replaces the messages it covers; pass
    False to size the conversation as the user sees it, in full.
    """
    system_parts = []
    if scenario is not None and scenario.context.strip():
        system_parts.append(scenario.context.strip())

    history = list(chat.messages)
    summary = chat.active_summary() if compacted else None
    if summary is not None:
        history = history[summary.covers:]
        system_parts.append(f"Context summary:\n{summary.text.strip()}")

    # Gemma-style templates raise on a leading assistant turn and no capability
    # flag reports it, so an opening line always moves into the prompt.
    lead_in = "Opening scene" if summary is None else "Most recent scene"
    while history and history[0].role is not Role.USER:
        opening = (history.pop(0).text or "").strip()
        if opening:
            system_parts.append(f"{lead_in}:\n{opening}")

    messages = []
    for m in history:
        if not (m.text or "").strip() and m.generation_error:
            continue
        messages.append({"role": str(m.role), "content": m.text or ""})
    if draft:
        messages.append({"role": str(Role.USER), "content": draft})

    return prompting.with_system(messages, "\n\n".join(system_parts),
                                 supports_system_role)


def opening_message(scenario: Scenario | None) -> Message | None:
    if scenario is None or not scenario.opening_text.strip():
        return None
    return Message(role=Role.ASSISTANT, name=scenario.name,
                   text=scenario.opening_text)
