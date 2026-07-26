"""The job model: progress, decisions, splitting, and serialisation."""

from tests.harness import check, section
from thaumaturgy.editing import Instructions, Job, Settings, Status
from thaumaturgy.editing.models import Budgets, reattach_edges

DOC = ("Para one line one. Line two.\n\nPara two here, a little longer than the "
       "first one was.\n\nPara three ends the piece.\n") * 30


def make(span_target=60, **settings) -> Job:
    job = Job(id="j1", title="doc", source_text=DOC,
              instructions=Instructions(), settings=Settings.from_dict(settings))
    job.budgets = Budgets(span_target=span_target, available=9999, overlap=0)
    job.divide()
    return job


def run() -> None:
    section("progress is measured in source characters")
    # Not spans: a truncated span is halved and retried, so the span count
    # climbs mid-run and a span-based bar slides backwards.
    job = make()
    check("starts at zero", job.progress()[0] == 0)
    check("total is the document", job.progress()[1] == len(DOC))
    check("percent starts at zero", job.percent() == 0)

    before_total = job.progress()[1]
    job.split(0)
    check("splitting leaves the total alone", job.progress()[1] == before_total)

    for span in job.spans[:3]:
        span.status = Status.ACCEPTED
    done = job.progress()[0]
    undecided = job.next_undecided()
    check("splitting an undecided span leaves progress alone",
          job.split(undecided) and job.progress()[0] == done)
    check("splitting a decided span is refused", job.split(0) is False)
    check("...and leaves its rewrite intact", job.spans[0].status is Status.ACCEPTED)

    for span in job.spans:
        span.status = Status.ACCEPTED
    d, t = job.progress()
    check("complete covers the whole document", d == t == len(DOC))
    check("percent reaches 100", job.percent() == 100)

    section("finding the next undecided span")
    job = make()
    check("first is zero", job.next_undecided() == 0)
    job.spans[0].status = Status.ACCEPTED
    job.spans[1].status = Status.ORIGINAL
    check("skips decided spans", job.next_undecided() == 2)
    check("searches after a cursor", job.next_undecided(2) == 3)
    check("a cursor past the end finds nothing",
          job.next_undecided(len(job.spans)) is None)
    for span in job.spans:
        span.status = Status.ACCEPTED
    check("none left when all decided", job.next_undecided() is None)

    section("assembling the document")
    job = make()
    check("undecided spans fall back to original", job.assemble() == DOC)
    job.spans[1].rewritten = "REPLACED. "
    job.spans[1].status = Status.ACCEPTED
    check("an accepted rewrite is used", "REPLACED." in job.assemble())
    job.spans[1].status = Status.ORIGINAL
    check("keep-original ignores the rewrite", "REPLACED." not in job.assemble())

    section("deciding restores boundary whitespace")
    # Models drop it, and a lost newline at a join welds two paragraphs.
    job = make()
    original = job.spans[1].original
    job.decide(1, Status.ACCEPTED, " ".join(original.split()))
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    check("leading whitespace restored", job.spans[1].rewritten.startswith(lead))
    check("trailing whitespace restored", job.spans[1].rewritten.endswith(trail))
    check("edges survive a flattening model",
          reattach_edges("Hello world.\n\n", "Hello, world.").endswith("\n\n"))
    check("nothing to restore is fine", reattach_edges("No edges", "No edges!") == "No edges!")

    section("splitting preserves the document")
    job = make(span_target=200)
    check("split reports success", job.split(0) is True)
    check("text intact after split", "".join(s.original for s in job.spans) == DOC)
    check("offsets stay contiguous",
          all(job.spans[i].end == job.spans[i + 1].start
              for i in range(len(job.spans) - 1)))
    tiny = Job(id="t", title="t", source_text="x", instructions=Instructions(),
               settings=Settings())
    tiny.spans = [__import__("thaumaturgy.editing.models", fromlist=["Span"]).Span(0, 1, "x")]
    check("an indivisible span refuses to split", tiny.split(0) is False)

    section("settings coercion survives a hand-edited file")
    check("garbage becomes defaults", Settings.from_dict("nonsense") == Settings())
    check("bad types fall back", Settings.from_dict({"temperature": "hot"}).temperature == 0.2)
    check("overlap clamped high", Settings.from_dict({"overlap_pct": 5}).overlap_pct == 0.45)
    check("overlap clamped low", Settings.from_dict({"overlap_pct": -1}).overlap_pct == 0.0)
    check("buffer cannot starve the span",
          Settings.from_dict({"max_new_tokens": 200,
                              "response_buffer": 9999}).response_buffer == 136)
    check("sampler params are the engine subset",
          set(Settings().sampler_params()) ==
          {"max_new_tokens", "temperature", "top_p", "top_k", "min_p", "repetition_penalty"})

    section("instructions coercion")
    check("garbage becomes defaults", Instructions.from_dict(None) == Instructions())
    check("blank strings are kept",
          Instructions.from_dict({"passage_instruction": ""}).passage_instruction == "")
    check("absent keys fall back",
          Instructions.from_dict({}).passage_instruction == Instructions().passage_instruction)
    check("non-strings are rejected, not crashed",
          Instructions.from_dict({"context_framing": 5}).context_framing
          == Instructions().context_framing)

    section("budgets")
    b = Budgets.derive(Settings.from_dict({"max_new_tokens": 700,
                                           "response_buffer": 150,
                                           "overlap_pct": 0.10}), 20000, 40)
    check("span target is under the reply cap", b.span_target == 550)
    check("the whole prompt fits the context",
          2 * b.span_target + 2 * b.overlap + 700 < 20000)
    check("zero overlap means no margin",
          Budgets.derive(Settings(), 20000, 40).overlap == 0)
    check("a tiny context still yields a usable budget",
          Budgets.derive(Settings(), 512, 40).span_target > 0)
    check("an unknown context limit is tolerated",
          Budgets.derive(Settings(), None, 40).span_target > 0)

    section("serialisation round-trip")
    job = make(span_target=200, overlap_pct=0.15, allow_deletions=True)
    job.instructions = Instructions(system_prompt="CUSTOM.", prime_reply=True,
                                    passage_instruction="P {passage}")
    job.decide(2, Status.ACCEPTED, "edited text")
    back = Job.from_dict(job.to_dict())
    check("settings survive", back.settings == job.settings)
    check("instructions survive", back.instructions == job.instructions)
    check("spans survive",
          [s.to_dict() for s in back.spans] == [s.to_dict() for s in job.spans])
    check("statuses survive as enums", back.spans[2].status is Status.ACCEPTED)
    check("source survives", back.source_text == DOC)
    check("assembly is unchanged", back.assemble() == job.assemble())
    check("status serialises as a readable string",
          job.to_dict()["spans"][2]["status"] == "accepted")
