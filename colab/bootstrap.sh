#!/usr/bin/env bash
# One command per Colab runtime: clone or fast-forward the repo, wire git push auth from the
# GH_TOKEN env var (set from Colab Secrets in the cell before), install, run the harness.
#   %%bash
#   bash <(curl -sL https://raw.githubusercontent.com/$GH_REPO/main/colab/bootstrap.sh)
set -euo pipefail
: "${GH_REPO:?set GH_REPO=owner/Tracer in the env cell}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "https://github.com/$GH_REPO.git" "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -n "${GH_TOKEN:-}" ]; then
  git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
  git config user.email "${GIT_EMAIL:-colab@vi-engine.local}"
  git config user.name  "${GIT_NAME:-vi-engine colab}"
fi
pip install -q -e ".[dev,ingest]" 2>/dev/null || pip install -q --break-system-packages -e ".[dev,ingest]"
make check
echo "bootstrap OK: $(git rev-parse --short HEAD) in $REPO_DIR"
