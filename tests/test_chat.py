"""Chat models, reply parsing, and prompt assembly."""

from tests.harness import check, section
from thaumaturgy.chat import Chat, Message, Role, Scenario, prompt, reply

SCENARIO = Scenario(name="Grondar", context="You are Grondar.",
                    opening_text="The tavern door swings open.")


def conversation() -> Chat:
    c = Chat(id="c1", scenario="Grondar")
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="The door swings."))
    c.append(Message(role=Role.USER, name="You", text="Hello."))
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="Well met."))
    return c


def run() -> None:
    section("reading a reply with channel markers")
    # Templates llama.cpp can't parse leave these in the text instead of
    # emitting reasoning events.
    check("no markers passes through",
          reply.split_channels("just a reply") == ("just a reply", ""))
    check("empty is safe", reply.split_channels("") == ("", ""))
    check("pipe dialect splits", reply.split_channels(
        "visible <|channel|>analysis<|message|>hidden")[1] == "hidden")
    check("newline dialect splits", reply.split_channels(
        "<|channel>thought\nhidden\n")[1] == "hidden")
    check("a half-streamed marker does not split",
          reply.split_channels("text <|channel>a") == ("text <|channel>a", ""))
    check("control tokens are stripped",
          "<|end|>" not in reply.split_channels(
              "<|channel|>final<|message|>done<|end|>")[0])
    check("thought channels are reasoning, others visible", reply.split_channels(
        "<|channel|>thinking<|message|>t<|channel|>final<|message|>v") == ("v", "t"))
    check("channel names are case-insensitive",
          reply.split_channels("<|channel|>Thought<|message|>t")[1] == "t")

    section("reasoning is promoted when there is nothing else")
    # Some models put ordinary prose in the thought channel and never open a
    # final one; the bubble would otherwise be empty.
    check("promoted when the reply is empty",
          reply.promote_reasoning("", "only thoughts") == ("only thoughts", ""))
    check("left alone when the reply has text",
          reply.promote_reasoning("visible", "thoughts") == ("visible", "thoughts"))
    # Stream reasoning and marker reasoning are joined; with no visible text
    # left, the combined reasoning is promoted into the reply.
    check("interpret joins both sources of reasoning, then promotes",
          reply.interpret("<|channel|>analysis<|message|>b", "a") == ("a\n\nb", ""))
    check("interpret keeps them apart when there is visible text",
          reply.interpret("v<|channel|>analysis<|message|>b", "a") == ("v", "a\n\nb"))

    section("messages")
    m = Message(role=Role.ASSISTANT, text="<|channel|>analysis<|message|>t")
    check("an assistant message resolves markers", m.display() == ("t", ""))
    u = Message(role=Role.USER, text="<|channel|>analysis<|message|>t")
    check("a user message is never split", u.display()[0] == u.text)
    check("is_user", u.is_user and not m.is_user)

    section("warnings explain a bad ending")
    check("a clean stop is silent",
          Message(role=Role.ASSISTANT, finish_reason="stop").warning() is None)
    check("no reason is silent", Message(role=Role.ASSISTANT).warning() is None)
    check("length blames the token cap", "Max new tokens" in
          Message(role=Role.ASSISTANT, finish_reason="length").warning())
    check("length blames the context when the budget is unrestricted",
          "context window" in Message(role=Role.ASSISTANT, finish_reason="length",
                                      finish_limit="context").warning())
    check("an error is reported", "failed" in
          Message(role=Role.ASSISTANT, generation_error="boom").warning())
    check("editing clears the generation state", _cleared())

    section("finding the reply that can be redone")
    c = conversation()
    check("the last assistant reply", c.latest_assistant_index() == 2)
    c.append(Message(role=Role.USER, text="more"))
    check("not when the user spoke last", c.latest_assistant_index() is None)
    opening_only = Chat(id="c2")
    opening_only.append(Message(role=Role.ASSISTANT, text="An opening."))
    check("not an opening line with no user turn",
          opening_only.latest_assistant_index() is None)

    section("prompt assembly")
    msgs = prompt.build(conversation(), SCENARIO, supports_system_role=True)
    check("system turn carries the scenario", msgs[0]["role"] == "system"
          and "You are Grondar." in msgs[0]["content"])
    # Gemma-style templates raise on a leading assistant turn, and no capability
    # flag reports it, so the opening always moves into the prompt.
    check("a leading assistant turn folds into the system prompt",
          "Opening scene" in msgs[0]["content"])
    check("history follows", [m["role"] for m in msgs[1:]] == ["user", "assistant"])
    check("a draft is appended",
          prompt.build(conversation(), SCENARIO, draft="D")[-1]["content"] == "D")

    no_sys = prompt.build(conversation(), SCENARIO, supports_system_role=False)
    check("without a system role it merges into the first user turn",
          no_sys[0]["role"] == "user" and "You are Grondar." in no_sys[0]["content"])
    # With no scenario there is still a system turn if an opening line has to
    # be moved out of the leading position.
    check("no scenario still relocates a leading assistant turn",
          prompt.build(conversation(), None)[0]["role"] == "system")
    plain = Chat(id="c4")
    plain.append(Message(role=Role.USER, text="hi"))
    check("nothing to relocate and no scenario means no system turn",
          prompt.build(plain, None)[0]["role"] == "user")

    failed = Chat(id="c3")
    failed.append(Message(role=Role.USER, text="hi"))
    failed.append(Message(role=Role.ASSISTANT, text="", generation_error="boom"))
    check("a failed empty reply is left out of the history",
          len(prompt.build(failed, None)) == 1)

    check("an opening message is built from the scenario",
          prompt.opening_message(SCENARIO).text == SCENARIO.opening_text)
    check("no opening text means no message",
          prompt.opening_message(Scenario(name="x")) is None)

    section("serialisation round-trip")
    c = conversation()
    c.messages[2].finish_reason = "length"
    c.messages[2].finish_limit = "context"
    c.messages[2].reasoning = "some thinking"
    back = Chat.from_dict(c.to_dict())
    check("messages survive",
          [m.to_dict() for m in back.messages] == [m.to_dict() for m in c.messages])
    check("roles survive as enums", back.messages[1].role is Role.USER)
    check("finish_limit survives", back.messages[2].finish_limit == "context")
    check("empty optional keys stay out of the file",
          "reasoning" not in c.messages[1].to_dict())
    check("a scenario loads from its stored shape",
          Scenario.from_dict({"name": "n", "context": "c", "_file": "slug"}).file == "slug")


def _cleared() -> bool:
    m = Message(role=Role.ASSISTANT, text="x", reasoning="r",
                finish_reason="length", generation_error="boom")
    m.clear_generation_state()
    return (m.finish_reason is None and m.generation_error is None
            and m.reasoning == "" and m.warning() is None)
