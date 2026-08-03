"""Chat models, reply parsing, and prompt assembly."""

import pytest

from thaumaturgy.chat import Chat, Message, Role, Scenario, prompt, reply

SCENARIO = Scenario(name="Grondar", context="You are Grondar.",
                    opening_text="The tavern door swings open.")


def conversation() -> Chat:
    c = Chat(id="c1", scenario="Grondar")
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="The door swings."))
    c.append(Message(role=Role.USER, name="You", text="Hello."))
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="Well met."))
    return c


# Templates llama.cpp can't parse leave these markers in the text instead of
# emitting reasoning events.
@pytest.mark.parametrize("text,expected", [
    ("just a reply", ("just a reply", "")),
    ("", ("", "")),
    ("visible <|channel|>analysis<|message|>hidden", ("visible", "hidden")),
    ("<|channel>thought\nhidden\n", ("", "hidden")),
    ("text <|channel>a", ("text <|channel>a", "")),  # half-streamed marker
    ("<|channel|>Thought<|message|>t", ("", "t")),  # names are case-insensitive
    ("<|channel|>thinking<|message|>t<|channel|>final<|message|>v", ("v", "t")),
])
def test_reading_a_reply_with_channel_markers(text, expected):
    assert reply.split_channels(text) == expected


def test_control_tokens_are_stripped():
    assert "<|end|>" not in reply.split_channels(
        "<|channel|>final<|message|>done<|end|>")[0]


def test_reasoning_is_promoted_when_there_is_nothing_else():
    # Some models put ordinary prose in the thought channel and never open a
    # final one; the bubble would otherwise be empty.
    assert reply.promote_reasoning("", "only thoughts") == ("only thoughts", "")
    assert reply.promote_reasoning("visible", "thoughts") == ("visible", "thoughts")
    assert reply.interpret("<|channel|>analysis<|message|>b", "a") == ("a\n\nb", "")
    assert reply.interpret("v<|channel|>analysis<|message|>b", "a") == ("v", "a\n\nb")


def test_only_an_assistant_message_resolves_markers():
    marked = "<|channel|>analysis<|message|>t"
    assert Message(role=Role.ASSISTANT, text=marked).display() == ("t", "")
    assert Message(role=Role.USER, text=marked).display()[0] == marked


def test_warnings_explain_a_bad_ending():
    assert Message(role=Role.ASSISTANT, finish_reason="stop").warning() is None
    assert "Max new tokens" in Message(role=Role.ASSISTANT,
                                       finish_reason="length").warning()
    assert "context window" in Message(role=Role.ASSISTANT, finish_reason="length",
                                       finish_limit="context").warning()
    assert "failed" in Message(role=Role.ASSISTANT, generation_error="boom").warning()


def test_editing_clears_the_generation_state():
    m = Message(role=Role.ASSISTANT, text="x", reasoning="r",
                finish_reason="length", generation_error="boom")
    m.clear_generation_state()
    assert m.finish_reason is None and m.generation_error is None
    assert m.reasoning == "" and m.warning() is None


def test_finding_the_reply_that_can_be_redone():
    c = conversation()
    assert c.latest_assistant_index() == 2
    c.append(Message(role=Role.USER, text="more"))
    assert c.latest_assistant_index() is None
    opening_only = Chat(id="c2")
    opening_only.append(Message(role=Role.ASSISTANT, text="An opening."))
    assert opening_only.latest_assistant_index() is None


def test_prompt_assembly():
    msgs = prompt.build(conversation(), SCENARIO, supports_system_role=True)
    assert msgs[0]["role"] == "system" and "You are Grondar." in msgs[0]["content"]
    # Gemma-style templates raise on a leading assistant turn, so the opening
    # always moves into the system prompt.
    assert "Opening scene" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"]
    assert prompt.build(conversation(), SCENARIO, draft="D")[-1]["content"] == "D"


def test_prompt_assembly_without_a_system_role_or_a_scenario():
    no_sys = prompt.build(conversation(), SCENARIO, supports_system_role=False)
    assert no_sys[0]["role"] == "user" and "You are Grondar." in no_sys[0]["content"]
    plain = Chat(id="c4")
    plain.append(Message(role=Role.USER, text="hi"))
    assert prompt.build(plain, None)[0]["role"] == "user"


def test_a_failed_empty_reply_is_left_out_of_the_history():
    failed = Chat(id="c3")
    failed.append(Message(role=Role.USER, text="hi"))
    failed.append(Message(role=Role.ASSISTANT, text="", generation_error="boom"))
    assert len(prompt.build(failed, None)) == 1


def test_the_opening_message_comes_from_the_scenario():
    assert prompt.opening_message(SCENARIO).text == SCENARIO.opening_text
    assert prompt.opening_message(Scenario(name="x")) is None


def test_serialisation_round_trip():
    c = conversation()
    c.messages[2].finish_reason = "length"
    c.messages[2].finish_limit = "context"
    c.messages[2].reasoning = "some thinking"
    back = Chat.from_dict(c.to_dict())
    assert [m.to_dict() for m in back.messages] == [m.to_dict() for m in c.messages]
    assert back.messages[1].role is Role.USER
    assert back.messages[2].finish_limit == "context"
    assert "reasoning" not in c.messages[1].to_dict()  # empty keys stay out
    assert Scenario.from_dict({"name": "n", "context": "c", "_file": "slug"}).file == "slug"
