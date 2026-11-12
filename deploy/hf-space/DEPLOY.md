# Deploying the live demo to Hugging Face Spaces

The live Space is a **Gradio** app (`deploy/hf-space/gradio_app.py`, copied to
`app.py` on push), not the FastAPI server used locally — a separate,
simpler front-end so the free-tier CPU/ZeroGPU Space needs no Docker image.
Any fix made to `lumen_rag/api/app.py` (the FastAPI app) does **not** apply
to the deployed Space unless the equivalent change is also made here.

`deploy/hf-space/README.md` (Space config via YAML frontmatter) and
`deploy/hf-space/requirements.txt` are Space-specific; `push.sh` swaps them
in on a staged copy without touching the real top-level README/requirements.

## One-time setup

1. Create the Space: https://huggingface.co/new-space
   - Owner: your HF username/org (e.g. `WickTech`)
   - Name: `lumen-rag`
   - SDK: **Gradio**
   - Visibility: Public
2. Authenticate locally: `hf auth login` (needs a token with **write**
   access, from https://huggingface.co/settings/tokens).
3. **Optional: set the `ADMIN_KEY` Space secret** (Settings → Variables and
   secrets → New secret) to a random value. Ingesting and resetting (the API
   routes and the Gradio callbacks) are refused on the Space while it is unset,
   so visitors cannot fill, poison or wipe the shared index; the bundled sample
   corpus is loaded at startup instead. With it set, the owner can ingest or
   reset by sending the same value (`X-Admin-Key` header, or the demo's
   admin-key field).

## Push a new version

Run from the repo root:

```bash
bash deploy/hf-space/push.sh
```

This uploads a staged copy of the repo (minus `deploy/` and `tests/`, with
`gradio_app.py` renamed to `app.py`) via the HF Hub `upload_folder()` API —
not `git push` and not `hf upload` (see comments in `push.sh` for why).

The Space rebuilds on push; first build takes a few minutes. Once live,
update the demo link in the main README/case study.
