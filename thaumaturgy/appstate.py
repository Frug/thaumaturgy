"""Process-global application state (single-user local app).

Kept deliberately simple — a module-level singleton, no UI imports (to avoid
import cycles). Pages read/write these to coordinate: e.g. the chat page reads
the currently-selected model when starting a new chat.
"""


class AppState:
    def __init__(self):
        self.current_model: str | None = None
        self.current_character: str | None = None
        self.current_params: dict = {}  # active sampler values for generation


state = AppState()
