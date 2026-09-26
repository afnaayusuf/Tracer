#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 23 build: choose the ReID embedder by measurement.
#   * tests/scenarios/warehouse_identities.yaml: which tube is which person, from the cast sheet
#   * bench/reid_eval.py: same-person vs different-person similarity for hist / SigLIP / DINOv2 /
#     OSNet (if torchreid installs), best threshold, and the two pairs that matter on this clip
#   * default agent model back to Qwen3.5-4B (fused kernels made it fast; it reads the cast better)
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_23.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 23"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] || warn "GH_TOKEN is not set: will commit locally but cannot push (use the one-cell form)"

step "0. runtime checks"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ "$HAS_GPU" = 1 ]; then echo "GPU: $(nvidia-smi -L | head -1)"; else warn "NO GPU in this runtime -> Runtime > Change runtime type > L4, then rerun"; fi
[ -f "$SOURCE" ] && echo "clip: $SOURCE ($(du -h "$SOURCE" | cut -f1))" || warn "NO CLIP at $SOURCE -> upload HI_DEF_VIDEO.mp4 to /content"

step "1. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin || true
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"
if [ -n "$(git status --porcelain)" ]; then
  warn "uncommitted changes from an interrupted run; discarding them (the script rewrites its files)"
  git checkout -- . && git clean -fdq
fi
git pull -q --ff-only 2>/dev/null || warn "pull skipped (offline or diverged); continuing on local HEAD"
[ -n "$(git log --grep='^session 22' --format=%h)" ] || die "session 22 commit not found; run build_session_22.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/cast_sheet.py
bench/reid_eval.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/scenario_eval.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/preflight.sh
colab/vllm_venv.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/scenarios/warehouse.yaml
tests/scenarios/warehouse_identities.yaml
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_reid.py
tests/test_schemas.py
tests/test_slice_gpu_smoke.py
tests/test_store_agent.py
tests/test_tubes.py
vi/__init__.py
vi/agent/__init__.py
vi/agent/loop.py
vi/agent/tools.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/fake.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
vi/episode/debug.py
vi/episode/keyframes.py
vi/episode/writer.py
vi/eval/__init__.py
vi/eval/mot.py
vi/events/__init__.py
vi/events/compiler.py
vi/events/zones.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/gate/heartbeat.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/reid/__init__.py
vi/reid/base.py
vi/schemas/__init__.py
vi/schemas/common.py
vi/schemas/contact_sheet.py
vi/schemas/episode.py
vi/schemas/event.py
vi/schemas/export.py
vi/schemas/fact.py
vi/schemas/scene_card.py
vi/schemas/tick.py
vi/schemas/tube.py
vi/store/__init__.py
vi/store/db.py
vi/store/loader.py
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/linker.py
vi/tubes/quality.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p tests/scenarios colab/sessions data/bench
cat > tests/scenarios/warehouse_identities.yaml << 'EOF_VI'
# Who each person tube is, read off the cast sheet (birth crops) and the clip. Keyed by birth time
# and the birth crop's top-left corner (stable across runs; tube ids are not once births change).
# sure: I can identify the person from the crop. unsure tubes are excluded from threshold selection.
clip: HI_DEF_VIDEO.mp4
people:
  P1: near-left of table, orange vest, black hair, facing table
  P2: near-left of table, dark jacket + orange vest, back to camera
  P3: far-left end of table, woman, orange vest over dark top
  P4: far side of table, white hard hat, orange vest, blue jeans
  P5: right of table at 0 s, pink/dark cap, orange vest, holding a bundle
  P6: conveyor, yellow-green hi-vis + white hard hat, present the whole clip
  P7: conveyor, yellow-green hi-vis, dark cap
  P8: dock aisle top-right, dark jacket + orange vest, walking with boxes
tubes:
  - {born_ms: 0,     x: 505,  y: 290, person: P1, sure: true}
  - {born_ms: 0,     x: 472,  y: 293, person: P2, sure: true}
  - {born_ms: 0,     x: 599,  y: 191, person: P3, sure: true}
  - {born_ms: 0,     x: 661,  y: 184, person: P4, sure: true}
  - {born_ms: 3667,  x: 712,  y: 207, person: P4, sure: true}
  - {born_ms: 5000,  x: 738,  y: 203, person: P4, sure: true}
  - {born_ms: 0,     x: 812,  y: 253, person: P5, sure: true}
  - {born_ms: 7000,  x: 1213, y: 246, person: P6, sure: true}
  - {born_ms: 4000,  x: 1215, y: 240, person: P7, sure: true}
  - {born_ms: 12667, x: 1215, y: 240, person: P8, sure: false}   # orange vest at the frame border; linked to P7 by SigLIP (wrong)
  - {born_ms: 3000,  x: 613,  y: 131, person: P3, sure: false}   # looks like the woman at the far end again
  - {born_ms: 4000,  x: 642,  y: 198, person: P5, sure: false}   # pink cap after moving to the far-middle
  - {born_ms: 5333,  x: 1183, y: 224, person: P7, sure: false}
  - {born_ms: 6000,  x: 1159, y: 204, person: P7, sure: false}
EOF_VI
cat > bench/reid_eval.py << 'EOF_VI'
"""Which appearance embedding actually tells these workers apart? For every embedder that loads,
embed each labelled tube's birth crop and compare same-person pairs against different-person
pairs. Prints one table and writes data/bench/reid.jsonl. Decides the linker's embedder/threshold.

  python bench/reid_eval.py data/episodes/*.jsonl --labels tests/scenarios/warehouse_identities.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from vi.episode import EpisodeWriter
from vi.reid import HistogramEmbedder
from vi.schemas.episode import TubeRecord


def load_embedders(names: list[str]) -> dict:
    out = {}
    for n in names:
        try:
            if n == "hist":
                out[n] = HistogramEmbedder()
            elif n == "siglip":
                from vi.reid.base import SigLIPEmbedder
                out[n] = SigLIPEmbedder()
            elif n == "dinov2":
                out[n] = DINOv2Embedder()
            elif n == "osnet":
                from vi.reid.base import OSNetEmbedder
                out[n] = OSNetEmbedder(model_name="osnet_x1_0")
            print(f"[reid_eval] {n}: loaded")
        except Exception as e:
            print(f"[reid_eval] {n}: unavailable ({type(e).__name__}: {str(e)[:100]})")
    return out


class DINOv2Embedder:
    """facebook/dinov2-small (Apache-2.0): self-supervised features, known to be strong for
    instance-level similarity; CLS token, L2-normalised."""

    name = "dinov2"
    dim = 384

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops):
        from PIL import Image
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.proc(images=ims, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        f = feats.float().cpu().numpy()
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--labels", default="tests/scenarios/warehouse_identities.yaml")
    ap.add_argument("--keyframes", default="data/keyframes")
    ap.add_argument("--embedders", default="hist,siglip,dinov2,osnet")
    ap.add_argument("--tol", type=int, default=25, help="px tolerance matching labels to birth crops")
    a = ap.parse_args()
    from PIL import Image
    labels = yaml.safe_load(Path(a.labels).read_text())
    tubes = [r.tube for r in EpisodeWriter.read(a.episode) if isinstance(r, TubeRecord) and r.tube.class_label == "person"]
    rows = []
    for t in tubes:
        ref = (t.keyframe_refs or [""])[0]
        if not ref.startswith("kf://"):
            continue
        name = Path(ref).stem                       # <born>_birth_<x>_<y>
        parts = name.split("_")
        try:
            born, x, y = int(parts[0]), int(parts[2]), int(parts[3])
        except Exception:
            continue
        lab = next((l for l in labels["tubes"] if abs(l["born_ms"] - born) <= 400 and abs(l["x"] - x) <= a.tol and abs(l["y"] - y) <= a.tol), None)
        path = Path(a.keyframes) / ref.replace("kf://", "")
        if lab and path.exists():
            rows.append({"tube": t.tube_id, "person": lab["person"], "sure": bool(lab.get("sure", True)),
                         "crop": np.asarray(Image.open(path).convert("RGB"))})
    print(f"labelled tubes with crops: {len(rows)} / {len(tubes)}  (sure: {sum(r['sure'] for r in rows)})")
    if len(rows) < 3:
        raise SystemExit("not enough labelled crops; check keyframes dir and label coordinates")
    embedders = load_embedders([e.strip() for e in a.embedders.split(",") if e.strip()])
    sure = [r for r in rows if r["sure"]]
    results = {}
    for name, emb in embedders.items():
        E = emb.embed([r["crop"] for r in rows])
        idx = {r["tube"]: i for i, r in enumerate(rows)}
        pos, neg = [], []
        for r1, r2 in itertools.combinations(sure, 2):
            s = float(E[idx[r1["tube"]]] @ E[idx[r2["tube"]]])
            (pos if r1["person"] == r2["person"] else neg).append(s)
        if not pos or not neg:
            continue
        ths = np.linspace(0.3, 0.99, 70)
        acc = [(np.mean([p >= th for p in pos]) + np.mean([n < th for n in neg])) / 2 for th in ths]
        best = float(ths[int(np.argmax(acc))])
        results[name] = {"pairs_same": len(pos), "pairs_diff": len(neg), "same_mean": round(float(np.mean(pos)), 3),
                         "same_min": round(float(np.min(pos)), 3), "diff_mean": round(float(np.mean(neg)), 3),
                         "diff_max": round(float(np.max(neg)), 3), "gap": round(float(np.min(pos) - np.max(neg)), 3),
                         "best_thr": round(best, 2), "balanced_acc": round(float(max(acc)), 3)}
        # the pairs that matter on this clip
        p4 = [r for r in sure if r["person"] == "P4"]
        results[name]["hardhat_pairs"] = [round(float(E[idx[x["tube"]]] @ E[idx[y["tube"]]]), 3) for x, y in itertools.combinations(p4, 2)]
        p7 = next((r for r in rows if r["person"] == "P7" and r["sure"]), None)
        p8 = next((r for r in rows if r["person"] == "P8"), None)
        if p7 and p8:
            results[name]["greenvest_vs_orangevest"] = round(float(E[idx[p7["tube"]]] @ E[idx[p8["tube"]]]), 3)
    print(f"\n{'embedder':8s} {'same mean/min':>14s} {'diff mean/max':>14s} {'gap':>6s} {'thr':>5s} {'bal.acc':>7s}  hard-hat pairs      green vs orange")
    for name, r in results.items():
        print(f"{name:8s} {r['same_mean']:>6.3f}/{r['same_min']:<6.3f} {r['diff_mean']:>6.3f}/{r['diff_max']:<6.3f} "
              f"{r['gap']:>6.3f} {r['best_thr']:>5.2f} {r['balanced_acc']:>7.3f}  {r['hardhat_pairs']}  {r.get('greenvest_vs_orangevest', '-')}")
    print("\nread: a positive gap means every same-person pair scores above every different-person pair; "
          "hard-hat pairs must clear the threshold, green-vs-orange must fall below it.")
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "reid.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "reid_eval", "labelled": len(rows), "results": results,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


if __name__ == "__main__":
    main()
EOF_VI
cp "$0" colab/sessions/session_23_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness (+ torchreid for OSNet, best effort)"
pipi -e ".[dev,ingest]"
if [ "$HAS_GPU" = 1 ]; then
  python -c "import fla" 2>/dev/null || pipi flash-linear-attention >/tmp/fla.log 2>&1 || true
  python -c "import torchreid" 2>/dev/null || { pipi torchreid gdown >/tmp/torchreid.log 2>&1 && echo "   torchreid installed" || warn "torchreid unavailable ($(tail -1 /tmp/torchreid.log | cut -c1-100)); OSNet row will be skipped"; }
fi
make check

step "4. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"; fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" 2>/dev/null; sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"; DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "5. slice -> store -> cast sheet"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes data/keyframes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|entities|relinks|max_concurrent_persons)\"" /tmp/slice_final.log | sed 's/^/  /'
  python bench/cast_sheet.py data/episodes/*.jsonl --keyframes data/keyframes --out data/bench/cast_sheet.jpg >/dev/null
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "6. ReID separability: which embedder tells these workers apart?"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  python bench/reid_eval.py data/episodes/*.jsonl --labels tests/scenarios/warehouse_identities.yaml --keyframes data/keyframes \
    --embedders hist,siglip,dinov2,osnet 2>&1 | grep -v "HF Hub\|Loading weights"
else
  warn "needs the real clip's keyframes; skipped"
fi

step "7. three questions, $AGENT_MODEL (transformers)"
BACKEND=fake; [ "$HAS_GPU" = 1 ] && BACKEND=transformers
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 2 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -E "^CAST|^Q:|latency:|A \(|clarify|cites:"

step "8. scenario acceptance, $AGENT_MODEL"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -E "^\s+\[(PASS|FAIL)\]|answered=|passed,"

step "9. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: identity labels from the cast sheet, ReID separability bench, 4B default"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "10. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste step 6 (the table) and step 8 back."
