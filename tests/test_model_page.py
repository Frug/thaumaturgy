import errno

from thaumaturgy.ui import model_page


def test_list_models_surfaces_directory_error(monkeypatch):
    error = PermissionError(errno.EACCES, "Permission denied", "/models/private")

    def fail_to_list_models():
        raise error

    monkeypatch.setattr(model_page.engine, "list_models", fail_to_list_models)

    models, reason = model_page._list_models()

    assert models == []
    assert reason == (
        "Can't read the models directory: "
        "[Errno 13] Permission denied: '/models/private'"
    )


def test_list_models_returns_available_models(monkeypatch):
    monkeypatch.setattr(model_page.engine, "list_models", lambda: ["model.gguf"])

    assert model_page._list_models() == (["model.gguf"], None)
