"""The job model: progress, decisions, splitting, and serialisation."""

from thaumaturgy.editing import Instructions, Job, Settings, Span, Status
from thaumaturgy.editing.models import Budgets, reattach_edges

DOC = ("Para one line one. Line two.\n\nPara two here, a little longer than the "
       "first one was.\n\nPara three ends the piece.\n") * 30


def make(span_target=60, **settings) -> Job:
    job = Job(id="j1", title="doc", source_text=DOC,
              instructions=Instructions(), settings=Settings.from_dict(settings))
    job.budgets = Budgets(span_target=span_target, available=9999, overlap=0)
    job.divide()
    return job


def test_progress_is_measured_in_source_characters():
    # Not spans: halving on retry grows the count, sliding the bar backwards.
    job = make()
    assert job.progress() == (0, len(DOC))
    assert job.percent() == 0

    job.split(0)
    assert job.progress()[1] == len(DOC)
    for span in job.spans[:3]:
        span.status = Status.ACCEPTED
    done = job.progress()[0]
    assert job.split(job.next_undecided())
    assert job.progress()[0] == done

    for span in job.spans:
        span.status = Status.ACCEPTED
    assert job.progress() == (len(DOC), len(DOC))
    assert job.percent() == 100


def test_a_decided_span_is_not_split():
    job = make()
    job.spans[0].status = Status.ACCEPTED
    assert job.split(0) is False
    assert job.spans[0].status is Status.ACCEPTED


def test_finding_the_next_undecided_span():
    job = make()
    assert job.next_undecided() == 0
    job.spans[0].status = Status.ACCEPTED
    job.spans[1].status = Status.ORIGINAL
    assert job.next_undecided() == 2
    assert job.next_undecided(2) == 3
    assert job.next_undecided(len(job.spans)) is None
    for span in job.spans:
        span.status = Status.ACCEPTED
    assert job.next_undecided() is None


def test_assembling_the_document():
    job = make()
    assert job.assemble() == DOC
    job.spans[1].rewritten = "REPLACED. "
    job.spans[1].status = Status.ACCEPTED
    assert "REPLACED." in job.assemble()
    job.spans[1].status = Status.ORIGINAL
    assert "REPLACED." not in job.assemble()


def test_deciding_restores_boundary_whitespace():
    # Models drop it, and a lost newline at a join welds two paragraphs.
    job = make()
    original = job.spans[1].original
    job.decide(1, Status.ACCEPTED, " ".join(original.split()))
    assert job.spans[1].rewritten.startswith(original[:len(original) - len(original.lstrip())])
    assert job.spans[1].rewritten.endswith(original[len(original.rstrip()):])
    assert reattach_edges("Hello world.\n\n", "Hello, world.").endswith("\n\n")


def test_splitting_preserves_the_document():
    job = make(span_target=200)
    assert job.split(0) is True
    assert "".join(s.original for s in job.spans) == DOC
    assert all(job.spans[i].end == job.spans[i + 1].start
               for i in range(len(job.spans) - 1))


def test_an_indivisible_span_refuses_to_split():
    job = Job(id="t", title="t", source_text="x", instructions=Instructions(),
              settings=Settings())
    job.spans = [Span(0, 1, "x")]
    assert job.split(0) is False


def test_settings_coercion_survives_a_hand_edited_file():
    assert Settings.from_dict("nonsense") == Settings()
    assert Settings.from_dict({"temperature": "hot"}).temperature == 0.2
    assert Settings.from_dict({"overlap_pct": 5}).overlap_pct == 0.45
    assert Settings.from_dict({"overlap_pct": -1}).overlap_pct == 0.0
    starved = Settings.from_dict({"max_new_tokens": 200, "response_buffer": 9999})
    assert starved.response_buffer == 136  # cannot starve the span


def test_instructions_coercion():
    assert Instructions.from_dict(None) == Instructions()
    assert Instructions.from_dict({"passage_instruction": ""}).passage_instruction == ""
    assert (Instructions.from_dict({"context_framing": 5}).context_framing
            == Instructions().context_framing)


def test_budgets():
    b = Budgets.derive(Settings.from_dict({"max_new_tokens": 700,
                                           "response_buffer": 150,
                                           "overlap_pct": 0.10}), 20000, 40)
    assert b.span_target == 550  # under the reply cap
    assert 2 * b.span_target + 2 * b.overlap + 700 < 20000  # prompt fits
    assert Budgets.derive(Settings(), 20000, 40).overlap == 0
    assert Budgets.derive(Settings(), 512, 40).span_target > 0
    assert Budgets.derive(Settings(), None, 40).span_target > 0  # unknown context


def test_serialisation_round_trip():
    job = make(span_target=200, overlap_pct=0.15, allow_deletions=True)
    job.instructions = Instructions(system_prompt="CUSTOM.", prime_reply=True,
                                    passage_instruction="P {passage}")
    job.decide(2, Status.ACCEPTED, "edited text")
    back = Job.from_dict(job.to_dict())
    assert back.settings == job.settings
    assert back.instructions == job.instructions
    assert [s.to_dict() for s in back.spans] == [s.to_dict() for s in job.spans]
    assert back.spans[2].status is Status.ACCEPTED
    assert back.source_text == DOC
    assert job.to_dict()["spans"][2]["status"] == "accepted"
