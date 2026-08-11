# Thaumaturgy

<img width="820" height="381" alt="banner" src="https://github.com/user-attachments/assets/b47d3aea-2ef6-49ec-8ffa-cd3ca949d9ad" />

Thaumaturgy is a local-LLM chat app for GGUF models. The UI is meant to be easy to use for beginners while still giving plenty of control for experimentation.

It's built with [NiceGUI](https://nicegui.io), model serving uses `llama.cpp`'s `llama-server`,
and package management is handled by [uv](https://docs.astral.sh/uv/).

Thaumaturgy started because I loved the simplicity of 
[textgen](https://github.com/oobabooga/text-generation-webui), but
I had a number of nits with the UI I wanted to improve. I wanted managing settings to be quicker and more intuitive (at least for me).
One local user, GGUF / `llama.cpp` models, scenarios,
chat history, sampler presets, and model runtime settings. User data is stored
as plain JSON/YAML files under the data directory. When a model is loaded,
thaumaturgy starts and manages the `llama-server` subprocess itself; a separate
server process does not need to be started first. Current text-generation-webui
also starts `llama-server` internally for its `llama.cpp` loader.

> **Status:** work in progress. Chat, scenario management, model loading, model
> downloading, safetensors-to-GGUF conversion, runtime profiles, and persisted
> sampler presets work today. Tools/MCP, the Notebook view, and the
> OpenAI/Anthropic API server are not yet ported.

## Running

```bash
uv sync              # install core deps (no torch)
make start           # uv run thaumaturgy         (no hot reload)
make start-dev       # python -m thaumaturgy.main (hot reload for development)
```

- `THAUM_PORT` — port to serve on (default `8080`).
- `THAUM_DATA` — data directory (default `./data`).
- `THAUM_LOG_DIR` — diagnostic log directory; overrides the one set on the Settings page, which is where it is normally configured (off by default).

The `training` extra (`uv sync --extra training`) adds torch/transformers/etc.,
used for the safetensors→GGUF conversion path in the model downloader and, later,
for LoRA training.

## Data Files

Everything user-owned lives under the data dir (`./data` by default) as plain
files: `scenarios/*.yaml`, `chats/*.json`, and `presets.yaml`. Default scenario
examples live in `thaumaturgy/defaults/scenarios/` and are copied into
`data/scenarios/` once for a fresh data directory. After that they are normal
local scenarios and can be edited or deleted in the UI. The whole `data/`
directory is gitignored, so those local changes do not show up as Git changes.

## Long Chats

A chat that outgrows the model's context window stops and offers to compact
itself: its oldest turns are summarized into a recap, and the model is sent that
recap in their place. Nothing is removed from the chat — the transcript stays
whole in `chats/*.json` and in the UI, alongside a `summaries` entry recording
which messages the recap stands for. Editing a message the recap covers retires
it, and the next reply offers to write a fresh one.

How long a recap may run is **Recap budget** in the parameter set on the Model
page (default 4000 tokens, capped at 15% of the context window so the recap
cannot crowd out the recent messages it made room for). It is a ceiling, not a
target: models stop when they judge the summary complete, so the recap is
usually shorter. The recap instructions live in `compaction.yaml` in the data
dir and can be edited; an unedited file picks up new defaults automatically.

The chat's Context panel breaks down what the model is being sent, jumps to the
point in the transcript where the recap takes over, shows the recap itself, and
compacts on demand — **Compact now** on an uncompacted chat, **Redo recap** to
write the current one again after changing the budget or the instructions.
Whether the chat draws a divider at that point is a Settings-page toggle.

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
