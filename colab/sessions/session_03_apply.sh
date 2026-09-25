#!/usr/bin/env bash
# Session 03: Ring 1 — RF-DETR through the vi/detect interface, ROI packing, ring1 bench (frame + roi).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-/content/Tracer}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
cd "$REPO_DIR"
if [ -n "$(git status --porcelain)" ]; then
  echo "repo has uncommitted changes; commit or stash them first:"; git status --short; exit 1
fi
echo "== applying session 03 payload to $REPO_DIR"
cp -R "$HERE/payload/." "$REPO_DIR/"
mkdir -p colab/sessions && cp "$HERE/apply.sh" colab/sessions/session_03_apply.sh
pip install -q -e ".[dev,ingest]" 2>/dev/null || pip install -q --break-system-packages -e ".[dev,ingest]"
echo "== make check"; make check
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "== GPU: $(nvidia-smi -L | head -1)"
  pip install -q rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    echo "== ring 1 bench, frame mode, nano + medium, 2 fps sampled"
    python bench/ring1_detect.py --source "$SOURCE" --models nano,medium --mode frame --fps 2 --frames 60
    echo "== ring 1 bench, roi mode, nano, batch 8"
    python bench/ring1_detect.py --source "$SOURCE" --models nano --mode roi --fps 2 --frames 60 --batch 8
  else
    echo "!! no clip at $SOURCE; upload one and rerun: SOURCE=/content/your.mp4 bash $0"
  fi
else
  echo "!! no GPU in this runtime; switch to L4 (Runtime > Change runtime type) and rerun"
fi
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "session 03: RF-DETR via vi/detect (verified 1.7.0 API), ROI packing, ring1 bench frame+roi, bench rows tracked"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push; fi
echo "== session 03 applied: $(git rev-parse --short HEAD)"
