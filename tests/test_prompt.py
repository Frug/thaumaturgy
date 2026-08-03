"""Assembling the messages for one span.

Splicing the passage into a context blob makes the model reproduce from the top
of the blob, and every word added around the passage is the author's to change.
"""

from thaumaturgy.editing import Instructions, Job, Settings, Status
from thaumaturgy.editing.models import Budgets
from thaumaturgy.editing.prompt import EDGE_WORDS, PromptBuilder

DOC = "".join(f"Sentence {n} carries topic {n} along for a while here. "
              for n in range(120))


def build(overlap=0, supports_system_role=True, **instr_kwargs):
    job = Job(id="t", title="t", source_text=DOC,
              instructions=Instructions(**instr_kwargs), settings=Settings())
    job.budgets = Budgets(span_target=60, available=9999, overlap=overlap)
    job.divide()
    return job, PromptBuilder(job, supports_system_role=supports_system_role)


def test_the_passage_arrives_alone_in_the_final_turn():
    job, pb = build()
    msgs = pb.messages(3)
    target = job.spans[3].original
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"].rstrip().endswith(target.rstrip())
    assert all(target not in m["content"] for m in msgs[:-1])
    assert all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:]))


def test_overlap_brings_the_neighbouring_text_into_the_prompt():
    _, pb = build(overlap=0)
    assert pb.context(3) == ("", "")
    _, pb = build(overlap=100)
    before, after = pb.context(3)
    assert before and after
    assert before[:40] in "".join(m["content"] for m in pb.messages(3))


def test_a_primed_reply_makes_it_a_real_exchange():
    _, pb = build(overlap=100)
    assert [m["role"] for m in pb.messages(3)] == ["system", "user"]
    _, pb = build(overlap=100, prime_reply=True, primed_reply="SPOKEN.")
    assert [m["role"] for m in pb.messages(3)] == ["system", "user", "assistant", "user"]
    assert pb.messages(3)[2]["content"] == "SPOKEN."


def test_edge_anchors_pin_the_ends_of_the_passage():
    job, pb = build()
    content = pb.messages(3)[-1]["content"]
    words = " ".join(job.spans[3].original.split()).split()
    assert " ".join(words[:EDGE_WORDS]) in content
    assert " ".join(words[-EDGE_WORDS:]) in content
    assert not any(p in content
                   for p in ("{first_words}", "{last_words}", "{passage}"))


def test_every_added_word_is_the_authors():
    job, pb = build(passage_instruction="", context_framing="", primed_reply="")
    msgs = pb.messages(3)
    assert msgs[-1]["content"] == job.spans[3].original
    assert [m["role"] for m in msgs] == ["system", "user"]
    job, pb = build(passage_instruction="FIX:\n{passage}\nEND.")
    assert pb.messages(3)[-1]["content"] == f"FIX:\n{job.spans[3].original}\nEND."
    _, pb = build(passage_instruction="use {curly} and {0}")
    assert "{curly}" in pb.messages(3)[-1]["content"]  # not a format specifier


def test_a_nudge_reaches_the_prompt():
    _, pb = build()
    assert "be terse" in pb.messages(3, "be terse")[-1]["content"]


def test_the_system_prompt_moves_when_there_is_no_system_role():
    job, pb = build(supports_system_role=False)
    msgs = pb.messages(3)
    assert msgs[0]["role"] == "user"
    assert Instructions().system_prompt in msgs[0]["content"]
    job.instructions = Instructions(system_prompt="")
    assert PromptBuilder(job, True).messages(3)[0]["role"] == "user"


def test_accepted_text_feeds_forward():
    job, pb = build(overlap=100)
    job.spans[2].rewritten = "REWRITTEN-SENTINEL. "
    job.spans[2].status = Status.ACCEPTED
    assert "REWRITTEN-SENTINEL" in "".join(m["content"] for m in pb.messages(3))
