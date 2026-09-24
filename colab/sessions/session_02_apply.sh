#!/usr/bin/env bash
# Session 02: ingest reader (PyAV), PTS/size guards, codec routing, Ring 0 bench, Colab tooling.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-/content/Tracer}"
cd "$REPO_DIR"
if [ -n "$(git status --porcelain)" ]; then
  echo "repo has uncommitted changes; commit or stash them first:"; git status --short; exit 1
fi
echo "== applying session 02 payload to $REPO_DIR"
cp -R "$HERE/payload/." "$REPO_DIR/"
mkdir -p colab/sessions && cp "$HERE/apply.sh" colab/sessions/session_02_apply.sh
pip install -q -e ".[dev,ingest]" 2>/dev/null || pip install -q --break-system-packages -e ".[dev,ingest]"
echo "== make check"; make check
echo "== ring 0 bench on a synthetic clip"; python bench/ring0_gate.py --synthetic
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "session 02: PyAV ingest reader, PTS/size guards, codec gate routing, ring0 bench, colab tooling"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push; fi
echo "== session 02 applied: $(git rev-parse --short HEAD)"
