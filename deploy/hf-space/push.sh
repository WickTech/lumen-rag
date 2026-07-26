#!/usr/bin/env bash
# Push the current repo to a Hugging Face Space, swapping in the
# Space-specific README (deploy/hf-space/README.md) without touching the
# real top-level README on this branch.
#
# Usage: ./deploy/hf-space/push.sh [space-remote-url]
# Requires: `huggingface-cli login` already run.
set -euo pipefail

SPACE_URL="${1:-https://huggingface.co/spaces/WickTech/lumen-rag}"
BRANCH="hf-space-deploy"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

git remote add hf-space "$SPACE_URL" 2>/dev/null || git remote set-url hf-space "$SPACE_URL"

git worktree add -f "/tmp/lumen-hf-space-deploy" HEAD
trap 'git worktree remove --force /tmp/lumen-hf-space-deploy' EXIT

cp deploy/hf-space/README.md /tmp/lumen-hf-space-deploy/README.md
cp deploy/hf-space/gradio_app.py /tmp/lumen-hf-space-deploy/app.py
cp deploy/hf-space/requirements.txt /tmp/lumen-hf-space-deploy/requirements.txt
cd /tmp/lumen-hf-space-deploy
git add README.md app.py requirements.txt
git -c user.name=deploy -c user.email=deploy@local commit -q -m "Space README + Gradio entrypoint" --allow-empty
git push -f hf-space "HEAD:refs/heads/main"

echo "Pushed to $SPACE_URL"
