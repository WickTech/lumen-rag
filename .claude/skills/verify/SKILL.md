---
name: verify
description: How to build and run lumen-rag to verify a change
---

# Verifying lumen-rag changes

No Chrome/CDP available in this WSL box (`which google-chrome chromium` → nothing) —
browser-harness will fail with `DevToolsActivePort not found`. Don't waste time on it.

## CLI / core engine
```
uv venv /tmp/.../scratchpad/.venv
uv pip install --python /tmp/.../scratchpad/.venv/bin/python -e .
source /tmp/.../scratchpad/.venv/bin/activate
lumen ingest data/docs && lumen ask "..." && lumen eval data/eval.jsonl --k 3
```

## Gradio app (deploy/hf-space/gradio_app.py)
No browser to drive it visually — instead exercise the exact same Blocks event
endpoints the UI buttons call, via `gradio_client` (same code path as a click,
not a unit-test import):

```
uv pip install --python .../venv/bin/python -e . gradio gradio_client
GRADIO_SERVER_PORT=7871 .../venv/bin/python deploy/hf-space/gradio_app.py &
python - <<'PY'
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7871/")
c.predict(api_name="/load_sample")
c.predict("question", 5, "hybrid", api_name="/ask")
c.predict(5, api_name="/run_eval")
c.predict(api_name="/reset_index")
c.predict([handle_file("path")], api_name="/upload_files")
PY
```
Endpoint names are the Python function names (`/load_sample`, `/ask`, `/run_eval`,
`/reset_index`, `/upload_files`) as declared by the `.click()`/`.upload()` wiring
in `gradio_app.py`. Kill the server after with `pkill -f gradio_app.py`.

## Deploy script (deploy/hf-space/push.sh)
Force-pushes to a real HF Space — destructive/irreversible, don't run it live.
Review by diff only; check README.md `app_file:` matches whatever push.sh copies
to `app.py` in the worktree.
