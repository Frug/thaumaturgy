import json

from thaumaturgy import engine


def test_full_response_log_records_only_generated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "log_dir", lambda: tmp_path)
    tally = engine._StreamTally(512, "auto", 0, False, True)

    tally.event('{"request_secret":"must not be copied"}', {"content": "Hello "})
    tally.event('{"another":"raw event"}', {"reasoning_content": "Think."})
    tally.event('{"more":"raw data"}', {"content": "world"})
    tally.finish_reason = "stop"
    tally.write()

    record = json.loads((tmp_path / "chat-responses.jsonl").read_text())
    assert record["content"] == "Hello world"
    assert record["reasoning_content"] == "Think."
    assert record["finish_reason"] == "stop"
    assert record["error"] is None
    assert "request_secret" not in record
    assert not (tmp_path / "chat-stream.log").exists()


def test_full_response_log_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "log_dir", lambda: tmp_path)
    tally = engine._StreamTally(512, "auto", 0, False, False)
    tally.event('{"content":"Hello"}', {"content": "Hello"})
    tally.write()

    assert not (tmp_path / "chat-responses.jsonl").exists()
