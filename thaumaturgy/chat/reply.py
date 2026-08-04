"""Reading a model's raw reply.

Some templates llama.cpp can't build a parser for (Gemma's) leave channel
markers in the text instead of emitting reasoning events, so the reply has to be
split here. Pure text in, text out.
"""

import re

# Matches both dialects: `<|channel>thought` and `<|channel|>analysis<|message|>`.
# The terminator is required so a name still streaming in ("<|channel>a") isn't
# read as complete.
_CHANNEL_MARKER = "<|channel"
_CHANNEL_RE = re.compile(
    r"<\|channel\|?>[ \t]*([A-Za-z0-9_.-]+)[ \t]*(?:<\|message\|>|<channel\|>|\r?\n)")
_CONTROL_RE = re.compile(
    r"<\|start\|>[ \t]*assistant|<\|(?:start|end|return|message)\|>|<channel\|>")
_THOUGHT_CHANNELS = {"thought", "thinking", "reasoning", "analysis"}


def join_blocks(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def split_channels(text: str) -> tuple[str, str]:
    """Split channel-marked output into visible text and reasoning text.

    Both halves are stripped: ui.markdown measures indentation from the first
    non-empty line and slices that many chars off every line, so a reply opening
    with llama.cpp's usual leading space loses a character per line below it.
    """
    if not text or _CHANNEL_MARKER not in text:
        return text.strip(), ""
    matches = list(_CHANNEL_RE.finditer(text))
    if not matches:
        return text.strip(), ""

    visible_parts = [_CONTROL_RE.sub("", text[:matches[0].start()])]
    reasoning_parts = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = _CONTROL_RE.sub("", text[match.end():end])
        target = (reasoning_parts if match.group(1).lower() in _THOUGHT_CHANNELS
                  else visible_parts)
        target.append(content)
    return join_blocks(visible_parts), join_blocks(reasoning_parts)


def promote_reasoning(text: str, reasoning: str) -> tuple[str, str]:
    """Promote reasoning to the reply when the model produced nothing else.

    Some models put ordinary prose in the thought channel and never open a final
    one; the bubble would otherwise be empty.
    """
    if text.strip():
        return text, reasoning
    return reasoning, ""


def interpret(raw_text: str, raw_reasoning: str) -> tuple[str, str]:
    """Turn accumulated stream output into what should be shown."""
    text, marker_reasoning = split_channels(raw_text)
    return promote_reasoning(text, join_blocks([raw_reasoning, marker_reasoning]))
