#!/usr/bin/env bash
# Push the current repo to a Hugging Face Space, swapping in the
# Space-specific README (deploy/hf-space/README.md) without touching the
# real top-level README on this branch.
#
# Usage: ./deploy/hf-space/push.sh [space-id]
# Requires: `hf auth login` already run.
#
# Uses HfApi().upload_folder() (the HTTP commit API), not `git push` or the
# `hf upload` CLI: git push needs a git-credential-compatible token (the
# `hf auth login` OAuth session token isn't one -> "Invalid username or
# password"), and `hf upload` unconditionally calls repos/create first, which
# 402s on a free-tier account even when the Space already exists.
set -euo pipefail

SPACE_ID="${1:-WickTech/lumen-rag}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

git archive HEAD | tar -x -C "$STAGE"
rm -rf "$STAGE/deploy" "$STAGE/tests"
cp deploy/hf-space/README.md "$STAGE/README.md"
cp deploy/hf-space/gradio_app.py "$STAGE/app.py"
cp deploy/hf-space/requirements.txt "$STAGE/requirements.txt"

python3 - "$SPACE_ID" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi

space_id, stage = sys.argv[1], sys.argv[2]
url = HfApi().upload_folder(
    repo_id=space_id,
    repo_type="space",
    folder_path=stage,
    commit_message="Deploy from push.sh",
    ignore_patterns=["**/__pycache__/**", ".git/**"],
)
print(f"Pushed to https://huggingface.co/spaces/{space_id} -> {url}")
PY
