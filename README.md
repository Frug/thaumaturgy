# thaumaturgy

A modern, portable, local-LLM chat app: an all-Python [NiceGUI](https://nicegui.io)
frontend over a GGUF / `llama.cpp` serving core. It rewrites the original's UI
into a format I find more intuitive, keeps user data portable (plain JSON/YAML —
no database), and installs cleanly with [uv](https://docs.astral.sh/uv/).

> **Status:** work in progress. Chat, character management, model loading, model
> downloading (with safetensors→GGUF conversion), and persisted parameter sets
> work today. Tools/MCP, the Notebook view, and the OpenAI/Anthropic API server
> are not yet ported.

## Running

```bash
uv sync              # install core deps (no torch)
make start           # uv run thaumaturgy         (no hot reload)
make start-dev       # python -m thaumaturgy.main (hot reload for development)
```

- `THAUM_PORT` — port to serve on (default `8080`).
- `THAUM_DATA` — data directory (default `./data`).

The `training` extra (`uv sync --extra training`) adds torch/transformers/etc.,
used for the safetensors→GGUF conversion path in the model downloader and, later,
for LoRA training.

## Data & portability

Everything user-owned lives under the data dir (`./data` by default) as plain
files: `characters/*.yaml`, `chats/*.json`, and `presets.yaml`. Defaults are
seeded from code on first run, and the whole `data/` directory is gitignored so
personal preferences and chats stay out of version control.

## Credits

thaumaturgy is a derivative work of
[**text-generation-webui**](https://github.com/oobabooga/text-generation-webui)
by **oobabooga** and its contributors, licensed under the AGPL-3.0. Portions of
the model-serving / engine layer are ported and adapted from that project; the
NiceGUI interface and the data/persistence layers are new. See [`NOTICE`](NOTICE)
for details.

## License

[GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later),
matching the license of the original project it derives from. Note the AGPL's
§13 network-use clause: if you run a modified version as a service others use
over a network, you must offer those users its source.
