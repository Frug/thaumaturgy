"""Process-global application state (single-user local app).

Kept deliberately simple — a module-level singleton, no UI imports (to avoid
import cycles). Pages read/write these to coordinate.
"""


class AppState:
    def __init__(self):
        self.current_model: str | None = None
        self.current_scenario: str | None = None
        self.current_chat_id: str | None = None
        self.current_params: dict = {}  # active sampler values for generation
        self.generations: dict[str, dict] = {}  # in-flight chat_id -> generation state


state = AppState()
