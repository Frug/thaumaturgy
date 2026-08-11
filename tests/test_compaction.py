"""Folding old turns into a recap, and what the model is sent afterwards."""

import pytest

from thaumaturgy import appstate, store
from thaumaturgy.chat import compaction, prompt
from thaumaturgy.chat.models import (Chat, Message, Role, Scenario, Summary,
                                     fingerprint)

SCENARIO = Scenario(name="Grondar", context="You are Grondar.",
                    opening_text="The tavern door swings open.")


def conversation(turns: int = 6, size: int = 40) -> Chat:
    c = Chat(id="c1", scenario="Grondar")
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="The door swings."))
    for i in range(turns):
        c.append(Message(role=Role.USER, name="You", text=f"user {i} " + "x" * size))
        c.append(Message(role=Role.ASSISTANT, name="Grondar",
                         text=f"reply {i} " + "y" * size))
    return c


def recap(c: Chat, covers: int, text: str = "Much happened.") -> Summary:
    s = Summary(text=text, covers=covers,
                fingerprint=fingerprint(c.messages[:covers]), tokens=4)
    c.summaries.append(s)
    return s


def test_a_recap_replaces_the_messages_it_covers():
    c = conversation()
    recap(c, 7)
    msgs = prompt.build(c, SCENARIO)
    assert "Context summary:" in msgs[0]["content"]
    assert "Much happened." in msgs[0]["content"]
    # Only the turns after the boundary are sent verbatim.
    assert len(msgs) == len(c.messages) - 7 + 1
    assert "user 0" not in str(msgs)
    assert "user 3" in str(msgs)


def test_the_full_transcript_is_still_available_uncompacted():
    c = conversation()
    recap(c, 7)
    full = prompt.build(c, SCENARIO, compacted=False)
    assert "user 0" in str(full)
    assert "Context summary" not in str(full)


def test_editing_a_covered_message_retires_the_recap():
    c = conversation()
    recap(c, 7)
    assert c.active_summary() is not None
    c.messages[2].text = "something else entirely"
    assert c.active_summary() is None
    # The chat still works; it just goes back to sending everything.
    assert "something else entirely" in str(prompt.build(c, SCENARIO))


def test_an_edit_falls_back_to_a_narrower_recap_that_still_holds():
    c = conversation()
    recap(c, 3, "early")
    recap(c, 9, "later")
    assert c.active_summary().text == "later"
    c.messages[5].text = "changed"       # inside the wide recap, outside the narrow
    assert c.active_summary().text == "early"


def test_a_recap_never_hides_the_scenario():
    c = conversation()
    recap(c, 7)
    assert "You are Grondar." in prompt.build(c, SCENARIO)[0]["content"]


def test_a_chat_that_fits_is_left_alone(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 100_000)
    assert compaction.plan(conversation(), SCENARIO) is None
    assert compaction.plan(None, SCENARIO) is None
    assert compaction.plan(Chat(id="empty"), SCENARIO) is None


def test_a_full_chat_plans_to_fold_its_oldest_turns(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 2048)
    c = conversation(turns=40, size=400)
    target = compaction.plan(c, SCENARIO)
    assert target is not None and target.possible
    assert target.start == 0
    assert 0 < target.covers < len(c.messages)
    # The retained tail opens on a user turn, so history reads as a reply.
    assert c.messages[target.covers].role is Role.USER


def test_re_compaction_starts_where_the_last_recap_stopped(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 2048)
    c = conversation(turns=40, size=400)
    first = compaction.plan(c, SCENARIO)
    recap(c, first.covers)
    for _ in range(10):
        c.append(Message(role=Role.USER, name="You", text="more " + "z" * 400))
        c.append(Message(role=Role.ASSISTANT, name="Grondar", text="on " + "w" * 400))
    second = compaction.plan(c, SCENARIO)
    assert second is not None
    assert second.start == first.covers
    assert second.covers > first.covers


def test_a_window_full_of_recent_turns_alone_cannot_be_compacted(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 512)
    c = Chat(id="big")
    c.append(Message(role=Role.USER, name="You", text="x" * 20_000))
    c.append(Message(role=Role.ASSISTANT, name="Grondar", text="y" * 20_000))
    target = compaction.plan(c, SCENARIO)
    assert target is not None and not target.possible


def test_writing_a_recap_records_what_it_stands_for(monkeypatch):
    monkeypatch.setattr(compaction, "_generate", lambda messages, budget: "A recap.")
    c = conversation(turns=20)
    target = compaction.Plan(start=0, covers=9, used=4000, total=4096, budget=400)
    summary = compaction.run(c, SCENARIO, target)
    assert summary.text == "A recap." and summary.covers == 9
    c.summaries.append(summary)
    assert c.active_summary() is summary


def test_the_summarizer_is_shown_the_previous_recap_and_only_the_new_turns(monkeypatch):
    c = conversation(turns=20)
    recap(c, 9, "Everything before.")
    seen = {}

    def capture(messages, budget):
        seen["ask"] = messages[-1]["content"]
        return "new recap"

    monkeypatch.setattr(compaction, "_generate", capture)
    target = compaction.Plan(start=9, covers=19, used=4000, total=8192, budget=400)
    summary = compaction.run(c, SCENARIO, target)
    assert summary.text == "new recap"
    assert "Everything before." in seen["ask"]
    assert "user 0" not in seen["ask"]   # already covered by the carried recap
    assert "user 8" in seen["ask"]


def test_an_empty_recap_is_refused(monkeypatch):
    monkeypatch.setattr(compaction, "_generate", lambda messages, budget: "  ")
    target = compaction.Plan(start=0, covers=5, used=1, total=4096, budget=400)
    with pytest.raises(RuntimeError):
        compaction.run(conversation(), SCENARIO, target)


def test_the_reply_allowance_comes_out_of_the_window(monkeypatch):
    monkeypatch.setattr(appstate.state, "current_params", {"max_new_tokens": 900})
    assert compaction.reserve() == 900
    monkeypatch.setattr(appstate.state, "current_params", {"max_new_tokens": "junk"})
    assert compaction.reserve() == 512


def test_recaps_survive_a_round_trip():
    c = conversation()
    recap(c, 7)
    back = Chat.from_dict(c.to_dict())
    assert back.active_summary().text == "Much happened."
    assert "summaries" not in conversation().to_dict()  # absent until there is one


def test_the_recap_instructions_are_editable_on_disk():
    doc = store.load_compaction_prompt()
    assert "{transcript}" in doc["instruction"]
    store.save_compaction_prompt({**doc, "system": "Be terse."})
    assert store.load_compaction_prompt()["system"] == "Be terse."
    store.save_compaction_prompt(doc)


def test_a_transcript_too_big_for_the_summarizer_is_trimmed_from_the_front():
    c = conversation(turns=30, size=2000)
    text = compaction._transcript(c.messages, limit=600)
    assert text.startswith("[") and "earlier turns omitted]" in text.split("\n")[0]
    assert "reply 29" in text          # the newest turns are the ones kept
    assert "user 0" not in text


def test_the_recap_budget_comes_from_the_parameter_set(monkeypatch):
    monkeypatch.setattr(appstate.state, "current_params", {"recap_tokens": 4000})
    assert compaction.recap_budget(55_000) == 4000        # the setting fits
    assert compaction.recap_budget(8_192) == 1228         # capped to 15% of a small window
    monkeypatch.setattr(appstate.state, "current_params", {})
    assert compaction.recap_budget(100_000) == compaction.RECAP_TOKENS_DEFAULT
    monkeypatch.setattr(appstate.state, "current_params", {"recap_tokens": "junk"})
    assert compaction.recap_budget(100_000) == compaction.RECAP_TOKENS_DEFAULT
    # A tiny window still gets a usable floor rather than a few dozen tokens.
    monkeypatch.setattr(appstate.state, "current_params", {"recap_tokens": 4000})
    assert compaction.recap_budget(1024) == compaction.MIN_RECAP_TOKENS


def test_a_bigger_recap_budget_keeps_less_verbatim_history(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 16_384)
    c = conversation(turns=120, size=400)
    monkeypatch.setattr(appstate.state, "current_params", {"recap_tokens": 256})
    small = compaction.plan(c, SCENARIO)
    monkeypatch.setattr(appstate.state, "current_params", {"recap_tokens": 4000})
    large = compaction.plan(c, SCENARIO)
    assert large.budget > small.budget
    assert large.covers > small.covers   # the recap's room comes out of the tail


def test_the_summarizer_is_told_the_span_and_a_length_floor(monkeypatch):
    seen = {}

    def capture(messages, budget):
        seen["ask"] = messages[-1]["content"]
        return "recap"

    monkeypatch.setattr(compaction, "_generate", capture)
    target = compaction.Plan(start=0, covers=9, used=4000, total=8192, budget=1000)
    compaction.run(conversation(turns=20), SCENARIO, target)
    assert "9 turns" in seen["ask"]            # how much it is condensing
    assert "450-750 words" in seen["ask"]      # a floor, not just a ceiling
    assert "{" not in seen["ask"]              # every placeholder was filled


def test_an_untouched_prompt_file_picks_up_a_new_default(tmp_path, monkeypatch):
    old = store._SUPERSEDED_COMPACTION["instruction"][0]
    store.save_compaction_prompt({**store._default_compaction_doc(), "instruction": old})
    assert store.load_compaction_prompt()["instruction"] != old
    # An edited file is the user's own and survives untouched.
    store.save_compaction_prompt({**store._default_compaction_doc(),
                                  "instruction": "My own wording. {transcript}"})
    assert store.load_compaction_prompt()["instruction"] == "My own wording. {transcript}"
    store.save_compaction_prompt(store._default_compaction_doc())


def test_an_edited_template_still_gets_the_transcript_but_not_stray_numbers():
    filled = compaction._fill("Just summarize it.",
                              {"turns": "9", "max_words": "750",
                               "recap": "", "transcript": "T: hello"})
    assert "T: hello" in filled     # the material is appended when unreferenced
    assert "9" not in filled and "750" not in filled


def test_compaction_can_be_forced_before_the_window_is_full(monkeypatch):
    monkeypatch.setattr(compaction, "window", lambda: 100_000)
    c = conversation(turns=40, size=400)
    assert compaction.plan(c, SCENARIO) is None
    forced = compaction.plan(c, SCENARIO, force=True)
    assert forced is not None and forced.possible
