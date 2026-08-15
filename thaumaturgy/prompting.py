"""Message assembly shared by the chat, compaction, and editing prompts."""


def with_system(messages: list[dict], system: str,
                supports_system_role: bool = True) -> list[dict]:
    """Put `system` in front of `messages`.

    A template with no system role takes it merged into the first user turn
    instead, since dropping it would lose the instructions entirely.
    """
    system = (system or "").strip()
    if not system:
        return messages
    if supports_system_role:
        return [{"role": "system", "content": system}, *messages]
    for message in messages:
        if message["role"] == "user":
            message["content"] = f"{system}\n\n{message['content']}".strip()
            return messages
    return [{"role": "user", "content": system}, *messages]
