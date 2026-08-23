from pathlib import Path

from thaumaturgy import engine, store


def test_models_directory_defaults_to_data_and_can_change_at_runtime(
        tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("THAUM_DATA", str(data))

    assert store.models_dir_setting() == ""
    assert engine.models_dir() == data / "models"
    assert engine.models_dir().is_dir()

    other_drive = tmp_path / "other-drive" / "ggufs"
    store.save_models_dir(str(other_drive))

    assert store.models_dir_setting() == str(other_drive)
    assert engine.models_dir() == other_drive
    assert other_drive.is_dir()

    (other_drive / "one.gguf").touch()
    assert engine.list_models() == ["one.gguf"]

    store.save_models_dir("  ")
    assert store.models_dir_setting() == ""
    assert engine.models_dir() == data / "models"
    assert "models_dir" not in store.load_app_config()


def test_metadata_cache_distinguishes_model_directories(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("THAUM_DATA", str(data))
    first = tmp_path / "disk-one"
    second = tmp_path / "disk-two"
    first.mkdir()
    second.mkdir()
    (first / "same.gguf").write_bytes(b"same-size")
    (second / "same.gguf").write_bytes(b"same-size")

    engine._ctx_cache.clear()
    monkeypatch.setattr(
        engine.metadata_gguf, "read_context_length",
        lambda path: 4096 if path.parent == first else 8192,
    )

    store.save_models_dir(str(first))
    assert engine.trained_ctx("same.gguf") == 4096
    store.save_models_dir(str(second))
    assert engine.trained_ctx("same.gguf") == 8192


def test_loaded_model_tracks_its_original_directory(tmp_path, monkeypatch):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_model = old_dir / "same.gguf"
    new_model = new_dir / "same.gguf"
    old_model.touch()
    new_model.touch()

    class LoadedServer:
        running = True
        model_path: Path | None = old_model

    monkeypatch.setattr(engine, "server", LoadedServer())
    monkeypatch.setattr(engine, "models_dir", lambda: new_dir)

    assert engine.delete_model("same.gguf") == ["same.gguf"]
    assert old_model.exists()
    assert not new_model.exists()
