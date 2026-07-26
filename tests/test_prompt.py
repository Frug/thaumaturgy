"""Assembling the messages for one span.

Layout is the highest-leverage thing in this feature: splicing the passage into
a context blob makes the model reproduce from the top of the blob instead of
the passage, and every word added around the passage is the author's to change.
"""

from tests.harness import check, section
from thaumaturgy.editing import Instructions, Job, Settings
from thaumaturgy.editing.models import Budgets
from thaumaturgy.editing.prompt import EDGE_WORDS, PromptBuilder

DOC = "".join(f"Sentence {n} carries topic {n} along for a while here. "
              for n in range(120))


def build(overlap=0, **instr_kwargs):
    job = Job(id="t", title="t", source_text=DOC,
              instructions=Instructions(**instr_kwargs), settings=Settings())
    job.budgets = Budgets(span_target=60, available=9999, overlap=overlap)
    job.divide()
    return job, PromptBuilder(job, supports_system_role=True)


def run() -> None:
    section("the passage arrives alone in the final turn")
    job, pb = build()
    msgs = pb.messages(3)
    target = job.spans[3].original
    check("system turn first", msgs[0]["role"] == "system")
    check("passage closes the final turn",
          msgs[-1]["content"].rstrip().endswith(target.rstrip()))
    check("passage is not spliced into an earlier turn",
          all(target not in m["content"] for m in msgs[:-1]))
    check("roles always alternate",
          all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:])))

    section("context")
    job, pb = build(overlap=0)
    before, after = pb.context(3)
    check("no overlap means no context", before == "" and after == "")
    check("window is the span alone", (pb.window(3).first, pb.window(3).last) == (3, 3))
    job, pb = build(overlap=100)
    before, after = pb.context(3)
    check("overlap brings neighbours in", before and after)
    w = pb.window(3)
    check("window brackets the span", w.first <= 3 <= w.last)
    check("context reaches the prompt", before[:40] in "".join(m["content"] for m in pb.messages(3)))

    section("a primed reply makes it a real exchange")
    job, pb = build(overlap=100)
    check("without priming, context folds into one user turn",
          [m["role"] for m in pb.messages(3)] == ["system", "user"])
    job, pb = build(overlap=100, prime_reply=True)
    check("with priming, roles alternate through four turns",
          [m["role"] for m in pb.messages(3)] == ["system", "user", "assistant", "user"])
    job, pb = build(overlap=100, prime_reply=True, primed_reply="SPOKEN.")
    check("the primed reply is the configured text", pb.messages(3)[2]["content"] == "SPOKEN.")

    section("edge anchors pin the ends of the passage")
    job, pb = build()
    content = pb.messages(3)[-1]["content"]
    words = " ".join(job.spans[3].original.split())
    check("first words quoted", " ".join(words.split()[:EDGE_WORDS]) in content)
    check("last words quoted", " ".join(words.split()[-EDGE_WORDS:]) in content)
    check("no placeholder survives",
          not any(p in content for p in ("{first_words}", "{last_words}", "{passage}")))

    section("every added word is the author's")
    job, pb = build(passage_instruction="", context_framing="", primed_reply="")
    msgs = pb.messages(3)
    check("blanking sends the passage and nothing else",
          msgs[-1]["content"] == job.spans[3].original)
    check("blanking leaves only system + passage",
          [m["role"] for m in msgs] == ["system", "user"])
    job, pb = build(passage_instruction="FIX:\n{passage}\nEND.")
    check("the passage placeholder positions the text",
          pb.messages(3)[-1]["content"] == f"FIX:\n{job.spans[3].original}\nEND.")
    job, pb = build(overlap=100, context_framing="CTX[{before}|{after}]")
    check("context placeholders substituted", "CTX[" in pb.messages(3)[-1]["content"])
    job, pb = build(passage_instruction="use {curly} and {0}")
    check("stray braces are not format specifiers",
          "{curly}" in pb.messages(3)[-1]["content"])

    section("nudges and the system role")
    job, pb = build()
    check("a nudge reaches the prompt", "be terse" in pb.messages(3, "be terse")[-1]["content"])
    job = Job(id="t", title="t", source_text=DOC,
              instructions=Instructions(), settings=Settings())
    job.budgets = Budgets(span_target=60, available=9999, overlap=0)
    job.divide()
    no_sys = PromptBuilder(job, supports_system_role=False).messages(3)
    check("without a system role the prompt merges into the first user turn",
          no_sys[0]["role"] == "user"
          and Instructions().system_prompt in no_sys[0]["content"])
    job.instructions = Instructions(system_prompt="")
    check("an empty system prompt adds no turn",
          PromptBuilder(job, True).messages(3)[0]["role"] == "user")

    section("accepted text feeds forward")
    from thaumaturgy.editing import Status
    job, pb = build(overlap=100)
    job.spans[2].rewritten = "REWRITTEN-SENTINEL. "
    job.spans[2].status = Status.ACCEPTED
    check("an accepted rewrite appears in the next span's context",
          "REWRITTEN-SENTINEL" in "".join(m["content"] for m in pb.messages(3)))
