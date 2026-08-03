"""Point the data directory somewhere disposable.

Unset, it resolves to ./data — the real chats, jobs, and llama-server pidfile.
"""

import os
import tempfile

os.environ.setdefault("THAUM_DATA", tempfile.mkdtemp(prefix="thaumaturgy-tests-"))
