# Deploying the live demo to Hugging Face Spaces

The app itself needs no changes to run as a Space — it's the same FastAPI
server + Dockerfile used locally, on port 8000. Only `deploy/hf-space/README.md`
(the Spaces config file, via YAML frontmatter) is Space-specific.

## One-time setup

1. Create the Space: https://huggingface.co/new-space
   - Owner: your HF username/org (e.g. `WickTech`)
   - Name: `lumen-rag`
   - SDK: **Docker**
   - Visibility: Public
2. Authenticate locally: `huggingface-cli login` (needs a token with **write**
   access, from https://huggingface.co/settings/tokens).

## Push a new version

Run from the repo root:

```bash
bash deploy/hf-space/push.sh
```

This pushes the current `HEAD` to the Space's git remote, using
`deploy/hf-space/README.md` as the Space's README (Spaces requires its config
frontmatter in the README at the repo root it sees, which is why the push
script swaps it in on a throwaway branch rather than touching the real
top-level README).

The Space rebuilds the Docker image on push; first build takes a few minutes.
Once live, update the demo link in the main README/case study.
