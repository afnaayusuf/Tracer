#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 18 build:
#   * vLLM server launched with a clean environment (no PYTHONPATH, LD_LIBRARY_PATH = driver dir
#     only) and a torch build matched to the runtime's torch (cu128 here); a mismatched venv is
#     rebuilt; a failed start prints its root-cause lines (the earlier replacement never applied)
#   * real zones for the warehouse clip (data/zones/HI_DEF_VIDEO.json): packing_table, conveyor,
#     dock_aisle, dock_doors (exit), left_racking (media -> the hanging jackets are dropped, E-DET-05)
#   * scenario rules ignore ids ("cam1:E11" is not the number 11)
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_18.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 18"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] || warn "GH_TOKEN is not set: will commit locally but cannot push (use the one-cell form)"

step "0. runtime checks"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ "$HAS_GPU" = 1 ]; then echo "GPU: $(nvidia-smi -L | head -1)"; echo "driver: $(nvidia-smi | grep -oE 'Driver Version: [0-9.]+|CUDA Version: [0-9.]+' | tr '\n' ' ')"
else warn "NO GPU in this runtime -> Runtime > Change runtime type > L4, then rerun"; fi
[ -f "$SOURCE" ] && echo "clip: $SOURCE ($(du -h "$SOURCE" | cut -f1))" || warn "NO CLIP at $SOURCE -> upload HI_DEF_VIDEO.mp4 to /content"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

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
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only 2>/dev/null || warn "pull skipped (offline or diverged); continuing on local HEAD"
[ -n "$(git log --grep='^session 17' --format=%h)" ] || die "session 17 commit not found; run build_session_17.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
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
mkdir -p data/zones tests/scenarios colab/sessions data/bench
cat > colab/vllm_venv.sh << 'EOF_VI'
#!/usr/bin/env bash
# vLLM in its own environment so its torch never touches the runtime's (Colab ships a CUDA-13 torch;
# vLLM wheels bring a CUDA-12.8 torch and torchaudio refuses to import next to it).
# Colab's python has no ensurepip, so the venv is built by, in order: uv -> venv --without-pip +
# pip bootstrap -> virtualenv. Each failure is printed and the next method is tried.
#   bash colab/vllm_venv.sh start [model] [port]   # install if needed, serve in background, wait
#   bash colab/vllm_venv.sh stop
#   bash colab/vllm_venv.sh status                  # up | down
#   bash colab/vllm_venv.sh venv                    # only build the venv (smoke test)
set -uo pipefail
VENV="${VLLM_VENV:-/content/vllm-venv}"
MODEL="${2:-${AGENT_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${3:-8000}"
LOG="/tmp/vllm.log"
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; echo "stopped"; exit 0; fi
if [ "$cmd" = status ]; then curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "up" || { echo "down"; exit 1; }; exit 0; fi
if [ "$cmd" = start ] && curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  served="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
  if [ "$served" = "$MODEL" ]; then echo "   vLLM already up: $served (reusing; run 'stop' to restart)"; exit 0; fi
  echo "   vLLM up with $served, restarting for $MODEL"; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; sleep 3
fi
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

make_venv() {
  [ -x "$VENV/bin/python" ] && return 0
  echo "-- venv via uv"
  if (command -v uv >/dev/null 2>&1 || pipi uv) && uv venv "$VENV" --python "$(command -v python3)" >/tmp/venv.log 2>&1 \
     && uv pip install --python "$VENV/bin/python" -q pip >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via python -m venv --without-pip + get-pip"
  if python3 -m venv --without-pip "$VENV" >/tmp/venv.log 2>&1 \
     && curl -sSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
     && "$VENV/bin/python" /tmp/get-pip.py -q >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via virtualenv"
  if pipi virtualenv && python3 -m virtualenv -q "$VENV" >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  return 1
}

driver_cuda() {  # e.g. 12.8 from nvidia-smi; the torch build must not be newer than this
  nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1
}

install_vllm() {
  if "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
    have="$("$VENV/bin/python" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)"
    main="$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)"
    if [ -n "$have" ] && [ -n "$main" ] && [ "$have" != "$main" ]; then
      echo "-- venv torch is cu$have but the runtime's torch is cu$main (its CUDA libs are on LD_LIBRARY_PATH): rebuilding to match"
      rm -rf "$VENV"; make_venv || return 1
    else
      return 0
    fi
  fi
  drv="$(driver_cuda)"
  main_cu="$(python3 -c "import torch; print((torch.version.cuda or '').replace('.', ''))" 2>/dev/null || true)"
  backend="${VLLM_TORCH_BACKEND:-${main_cu:+cu$main_cu}}"; backend="${backend:-auto}"
  echo "-- installing vllm into the venv (driver CUDA ${drv:-?}, torch backend $backend; 3-6 min)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q --torch-backend="$backend" vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv, torch-backend=$backend)"; return 0; fi
  echo "   uv with --torch-backend failed: $(tail -1 /tmp/vllm_install.log | cut -c1-120)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv)"; return 0; fi
  if "$VENV/bin/python" -m pip install -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (pip)"; return 0; fi
  echo "   failed: $(tail -2 /tmp/vllm_install.log | tr '\n' ' ')"; return 1
}

root_cause() {  # the informative lines of a failed vLLM start, not its last twelve
  grep -iE "error|cuda|driver|out of memory|no kernel image|not supported|Traceback" "$LOG" | grep -v "TracerWarning" | head -12 | cut -c1-200
}

make_venv || { echo "could not create a virtualenv by any method"; exit 1; }
if [ "$cmd" = venv ]; then "$VENV/bin/python" -c "import sys; print('   venv python', sys.version.split()[0])"; exit 0; fi
install_vllm || exit 1
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1
echo "   driver CUDA $(driver_cuda)"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
# clean process environment: no PYTHONPATH, no user site, and only the driver's library dir on LD_LIBRARY_PATH
# (Colab's LD_LIBRARY_PATH points at the main environment's CUDA libraries, which can shadow the venv's)
nohup env -u PYTHONPATH PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/usr/lib64-nvidia}" \
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
  --max-model-len 16384 --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.6}" --dtype bfloat16 --max-num-seqs 4 \
  --enable-prefix-caching > "$LOG" 2>&1 &
echo "-- waiting for http://127.0.0.1:$PORT (weights download on first run)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   vLLM up: $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; exit 0
  fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "   server exited; root cause:"; root_cause; exit 1; fi
  sleep 5
done
echo "   server did not come up in 10 min; root cause:"; root_cause; exit 1
EOF_VI
cat > data/zones/HI_DEF_VIDEO.json << 'EOF_VI'
[
  {"zone_id": "packing_table", "camera_id": "cam1", "tile_id": "floor", "kind": "generic",
   "polygon": [[455, 175], [775, 175], [775, 535], [455, 535]]},
  {"zone_id": "conveyor", "camera_id": "cam1", "tile_id": "floor", "kind": "generic",
   "polygon": [[1095, 230], [1280, 230], [1280, 460], [1095, 460]]},
  {"zone_id": "dock_aisle", "camera_id": "cam1", "tile_id": "floor", "kind": "generic",
   "polygon": [[1040, 115], [1280, 115], [1280, 232], [1040, 232]]},
  {"zone_id": "dock_doors", "camera_id": "cam1", "tile_id": "floor", "kind": "exit",
   "polygon": [[370, 0], [560, 0], [560, 115], [370, 115]]},
  {"zone_id": "left_racking", "camera_id": "cam1", "tile_id": "floor", "kind": "media",
   "polygon": [[0, 95], [145, 95], [145, 335], [0, 335]]}
]
EOF_VI
cat > bench/slice_gpu.py << 'EOF_VI'
"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source clip.mp4 --detect hybrid      # ROIs + 1 Hz full-frame heartbeat (default)
  python bench/slice_gpu.py --source clip.mp4 --detect roi         # motion ROIs only (session 04/05 behaviour)
  python bench/slice_gpu.py --source clip.mp4 --detect frame       # full frame every tick (upper bound on recall)
  python bench/slice_gpu.py --source clip.mp4 --zones data/zones/cam1.json --tile lobby
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi.detect import (blobs_to_rois, crop_roi, dedupe_detections, full_frame_roi, is_tube_class, pad_batch,
                       remap_detections)
try:
    from vi.detect.rfdetr import RFDETRDetector
except ImportError:  # CPU runtime: only --model fake works
    RFDETRDetector = None  # type: ignore
from vi.episode import EpisodeWriter, KeyframeStore, annotate
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate, HeartbeatScheduler
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.reid import crop_for_embedding, make_embedder
from vi.tubes import TRACKERS, TubeLinker, grade_tube


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--tile", default="tile1")
    ap.add_argument("--model", default="nano")
    ap.add_argument("--fps", type=float, default=4.0, help="R11 range 2-5; IoU-only tracking collapses below ~4")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="detector floor; ByteTracker re-attaches 0.1-0.5 detections and births only >=0.5")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="byte")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from timing (JIT profiling runs)")
    ap.add_argument("--detect", choices=["roi", "frame", "hybrid"], default="frame",
                    help="frame (default, measured best at one camera): full frame every tick; roi: motion ROIs only; "
                         "hybrid: ROIs + full-frame heartbeat (multi-camera cost lever, under investigation)")
    ap.add_argument("--heartbeat-ms", type=int, default=1000, help="hybrid: full-frame detection period on active tiles")
    ap.add_argument("--quiet-heartbeat-ms", type=int, default=10000)
    ap.add_argument("--debug-frames", type=int, default=0, help="save N annotated frames to data/debug/<episode>/")
    ap.add_argument("--reid", choices=["none", "auto", "hist", "siglip", "osnet"], default="none",
                    help="appearance embeddings + TubeLinker (E-TUBE-04); auto = osnet > siglip > hist")
    ap.add_argument("--reid-every-ticks", type=int, default=8, help="gallery refresh cadence for active tubes")
    ap.add_argument("--reid-sim", type=float, default=0.75)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default: data/zones/<clip stem>.json if present, else edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    batch = 1 if a.detect == "frame" else a.batch          # frame mode: one image per tick, trace at 1
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=batch)   # CPU smoke path (tests, no GPU)
    else:
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=batch)
    gate = FrameDiffGate(a.camera)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    current = {"frame": None}
    zones = None
    tracker = compiler = writer = ep = None
    stage = Counter()
    detect_ms: list[float] = []
    ev_types: Counter = Counter()
    births = 0
    closed_all = []
    tick_ms = int(1000 / a.fps)
    frames = 0
    hb = HeartbeatScheduler(active_ms=a.heartbeat_ms, quiet_ms=a.quiet_heartbeat_ms)
    heartbeats = 0
    births_by_origin: Counter = Counter()
    confirmed_by_origin: Counter = Counter()
    rebirths = 0                                   # tube born next to one that just died = fragmentation, counted directly
    recent_dead: list[tuple[int, float, float, float]] = []   # (t_ms, cx, cy, h)
    seen_ids: set[str] = set()
    confirmed_ids: set[str] = set()
    origin_of: dict[str, str] = {}
    debug_every = max(1, int(a.fps * 1.5)) if a.debug_frames else 0     # one frame every ~1.5 s until N saved
    duplicate_pairs: list[dict] = []                # two live person tubes on one body: the hybrid bug, caught in the act
    dup_frames: list[str] = []
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    linker = TubeLinker(a.camera, sim_thr=a.reid_sim) if embedder else None
    embed_ms: list[float] = []
    last_embed_tick: dict[str, int] = {}
    pending_link: dict[str, np.ndarray] = {}
    relink_events = 0
    debug_paths: list[str] = []
    person_dets: list[int] = []
    concurrent_persons: list[int] = []
    state_ticks: Counter = Counter()

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones_path = a.zones or (str(Path("data/zones") / (Path(a.source).stem + ".json")) if (Path("data/zones") / (Path(a.source).stem + ".json")).exists() else None)
            zones = load_zones(zones_path, a.camera) if zones_path else default_zones(a.camera, w, h, tile_id=a.tile)
            media_zones = [z for z in zones if z.kind == "media"]
            print(f"zones: {[z.zone_id for z in zones]} ({'file ' + zones_path if zones_path else 'defaults'})")
            exit_boxes = [z.polygon for z in zones if z.kind == "exit"]
            from vi.schemas import Box
            exits = [Box(x1=min(p[0] for p in poly), y1=min(p[1] for p in poly),
                         x2=max(p[0] for p in poly), y2=max(p[1] for p in poly)) for poly in exit_boxes]
            tkw = {"iou_thr": a.iou_thr} if a.tracker == "simple" else {}
            tracker = TRACKERS[a.tracker](a.camera, max_occluded_ms=a.max_occluded_ms, exit_boxes=exits,
                                          keyframe_sink=kf.make_sink(lambda: current["frame"]), **tkw)
            compiler = EventCompiler(a.camera, zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=a.tile)
            writer = EpisodeWriter(a.out)
            ep = writer.open(a.tile, [a.camera], CamTime(cam_utc_ms=fr.pts_ms), prov)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        stage["gate"] += time.perf_counter() - t0
        events = compiler.on_gate(g)
        t0 = time.perf_counter()
        det_source = "detector"
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride) if a.detect in ("roi", "hybrid") else []
        tile_active = len(tracker._tracks) > 0
        if a.detect == "frame" or (a.detect == "hybrid" and hb.due(fr.pts_ms, tile_active)):
            rois.append(full_frame_roi(w, h))                   # heartbeat = one more crop in the batch
            det_source = "heartbeat"
            heartbeats += 1
        dets = []
        for i in range(0, len(rois), batch):
            chunk = rois[i:i + batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = dedupe_detections([d for d in dets if is_tube_class(d.class_label)])   # E-DET-10
        if media_zones:   # E-DET-05: jackets on a rack, posters, screens are not people
            dets = [d for d in dets if not any(z.contains(d.box.foot_point()) for z in media_zones)]
        person_dets.append(sum(1 for d in dets if d.class_label == "person" and d.confidence >= 0.5))
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms, det_source=det_source)
        # provenance + rebirth accounting (person tubes only)
        for t in live:
            if t.class_label != "person":
                continue
            if t.tube_id not in seen_ids:
                seen_ids.add(t.tube_id)
                origin = next((d.origin for d in dets if d.box.iou(t.box) > 0.7), "unknown")
                births_by_origin[origin] += 1
                origin_of[t.tube_id] = origin
                cx, cy = (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2
                if any(fr.pts_ms - tm <= 1500 and ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 <= hh for tm, x, y, hh in recent_dead):
                    rebirths += 1
            if t.state.value == "active" and t.tube_id not in confirmed_ids:
                confirmed_ids.add(t.tube_id)
                confirmed_by_origin[origin_of.get(t.tube_id, "unknown")] += 1
        if linker is not None:
            # embed at first sight, but only *link* once the tracker has confirmed the tube (state
            # active): an unconfirmed tube can still be deleted next tick, and a link to a tube
            # that never existed corrupts the cast (seen in session 12).
            due_birth = [t for t in live if t.class_label == "person" and t.tube_id not in last_embed_tick]
            due_refresh = [t for t in live if t.class_label == "person" and t.state.value == "active"
                           and t.tube_id in last_embed_tick and t.tube_id not in pending_link
                           and frames - last_embed_tick[t.tube_id] >= a.reid_every_ticks]
            todo = due_birth + due_refresh
            if todo:
                t0 = time.perf_counter()
                embs = embedder.embed([crop_for_embedding(fr.rgb, t.box) for t in todo])
                embed_ms.append((time.perf_counter() - t0) * 1000)
                birth_ids = {x.tube_id for x in due_birth}
                for t, e in zip(todo, embs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in birth_ids:
                        pending_link[t.tube_id] = e
                    else:
                        linker.on_refresh(t, e)
            live_ids = {t.tube_id for t in live}
            for t in live:
                if t.tube_id in pending_link and t.state.value == "active":
                    ev = linker.on_birth(t, pending_link.pop(t.tube_id), fr.pts_ms)
                    if ev is not None:
                        events.append(ev)
                        relink_events += 1
                        print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']}")
            for tid in [k for k in pending_link if k not in live_ids]:
                pending_link.pop(tid)                       # deleted while unconfirmed: never linked
            for t in live:
                if t.class_label == "person":
                    t.entity_id = linker.entity_of(t.tube_id)
            for t in closed:
                linker.on_close(t, fr.pts_ms)
        persons = [t for t in live if t.class_label == "person" and t.state.value in ("active", "born")]
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                ti, tj = persons[i], persons[j]
                iou = ti.box.iou(tj.box)
                if iou >= 0.5 and len(duplicate_pairs) < 12:
                    dets_here = [{"origin": d.origin, "conf": round(d.confidence, 2), "box": [round(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                                  "iou_i": round(d.box.iou(ti.box), 2), "iou_j": round(d.box.iou(tj.box), 2)}
                                 for d in dets if d.class_label == "person" and (d.box.iou(ti.box) > 0.1 or d.box.iou(tj.box) > 0.1)]
                    duplicate_pairs.append({"t_ms": fr.pts_ms, "hb": det_source == "heartbeat", "iou": round(iou, 2),
                                            "a": {"id": ti.tube_id, "origin": origin_of.get(ti.tube_id), "state": ti.state.value,
                                                  "box": [round(v) for v in (ti.box.x1, ti.box.y1, ti.box.x2, ti.box.y2)]},
                                            "b": {"id": tj.tube_id, "origin": origin_of.get(tj.tube_id), "state": tj.state.value,
                                                  "box": [round(v) for v in (tj.box.x1, tj.box.y1, tj.box.x2, tj.box.y2)]},
                                            "dets_on_tick": dets_here})
                    if len(dup_frames) < 4:
                        p = annotate(fr.rgb, rois, dets, live, f"DUPLICATE t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                                     Path(a.out).parent / "debug" / ep / f"dup_{frames:04d}.jpg")
                        if p:
                            dup_frames.append(str(p))
        for t in closed:
            if t.class_label == "person":
                recent_dead.append((fr.pts_ms, (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2, t.box.height))
        recent_dead = [r for r in recent_dead if fr.pts_ms - r[0] <= 1500]
        if debug_every and frames % debug_every == 0 and len(debug_paths) < a.debug_frames:
            p = annotate(fr.rgb, rois, dets, live, f"t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                         Path(a.out).parent / "debug" / ep / f"tick_{frames:04d}.jpg")
            if p:
                debug_paths.append(str(p))
        concurrent_persons.append(sum(1 for t in live if t.class_label == "person" and t.state.value == "active"))
        state_ticks.update(t.state.value for t in live if t.class_label == "person")
        births += max(0, len(live) + len(closed) - before) if closed else max(0, len(live) - before)
        closed_all += closed
        stage["track"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                              det_source="detector" if t.state.value == "active" else "predicted") for t in live]
        events += compiler.on_tick(snaps, fr.pts_ms)
        tick = Tick(camera_id=a.camera, tile_id=a.tile, tick_index=frames, t_start=CamTime(cam_utc_ms=fr.pts_ms),
                    t_end=CamTime(cam_utc_ms=fr.pts_ms + tick_ms), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            ev_types[e.type.value] += 1
            print(f"t={fr.pts_ms:7d}  {e.type.value:22s} zone={e.zone_id} subjects={e.subject_tube_ids}")
        stage["compile"] += time.perf_counter() - t0
        frames += 1

    if frames == 0:
        raise SystemExit("no frames read; check --source")
    tubes = [tr.tube for tr in tracker._tracks.values()] + closed_all
    cast = [CastMember(tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0])
            for t in tubes]
    for t in tubes:
        grade_tube(t, w, h)
        writer.write_tube(ep, t)
    writer.close(ep, CamTime(cam_utc_ms=fr.pts_ms + tick_ms), EpisodeStatus.closed, cast)
    st = reader.stats
    row = {
        "ring": "slice", "source": Path(a.source).name, "model": f"rf-detr-{a.model}", "sampled_fps": a.fps,
        "tracker": a.tracker, "detect": a.detect, "det_threshold": a.threshold, "heartbeats": heartbeats,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
        "person_tubes_low_quality": sum(1 for t in tubes if t.class_label == "person" and t.quality == "low"),
        "person_dets_per_frame": round(float(np.mean(person_dets)), 2) if person_dets else 0,
        "max_concurrent_persons": max(concurrent_persons) if concurrent_persons else 0,
        "person_visibility_duty": round(state_ticks["active"] / max(1, state_ticks["active"] + state_ticks["occluded"]), 3),
        "fragmentation_est": round(sum(1 for t in tubes if t.class_label == "person") / max(1.0, float(np.mean(person_dets))), 2) if person_dets else None,
        "reid": embedder.name if embedder else "none",
        "entities": linker.entities if linker else None, "relinks": linker.relinks if linker else None,
        "entities_per_concurrent": round(linker.entities / max(1, max(concurrent_persons) if concurrent_persons else 1), 2) if linker else None,
        "embed_ms_p50": round(float(np.median(embed_ms)), 2) if embed_ms else None,
        "births_by_origin": dict(births_by_origin), "confirmed_by_origin": dict(confirmed_by_origin),
        "rebirths": rebirths, "duplicate_pairs": duplicate_pairs, "duplicate_frames": dup_frames,
        "debug_frames": debug_paths,
        "tubes_per_concurrent": round(sum(1 for t in tubes if t.class_label == "person") / max(1, max(concurrent_persons) if concurrent_persons else 1), 2),
        "mean_person_tube_life_s": round(float(np.mean([(t.last_seen.corrected_ms() - t.born.corrected_ms()) / 1000
                                                        for t in tubes if t.class_label == "person"] or [0])), 2),
        "merge_candidates_flagged": sum(1 for t in tubes if t.merge_candidates),
        "tubes_by_final_state": dict(Counter(t.state.value for t in tubes)),
        "classes": dict(Counter(t.class_label for t in tubes)),
        "events": dict(ev_types), "keyframes_saved": kf.saved, "episode_id": ep,
        "episode_records": sum(1 for _ in EpisodeWriter.read(writer.path(ep))),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "slice_gpu.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}" + (f"\ndebug frames -> {Path(a.out).parent / 'debug' / ep}" if debug_paths else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/scenario_eval.py << 'EOF_VI'
"""Score the agent against a ground-truth scenario file. Every question gets pass/fail on the
mention rules, the citation rules and the latency budget; the row goes to data/bench/scenario.jsonl.

  python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend openai
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, search_events
from vi.agent.loop import ID_RE
from vi.store import connect


def _mentioned(rule, text: str) -> bool:
    """a string must appear; a list means any of its strings must appear"""
    if isinstance(rule, list):
        return any(str(r).lower() in text for r in rule)
    return str(rule).lower() in text


def score(res: dict, spec: dict, budget_ms: int) -> dict:
    f = res["final"]
    text = ID_RE.sub(" ", f.get("text") or f.get("question") or "").lower()    # ids are not numbers
    checks = {
        "answered": f["action"] == "answer",
        "mentions": all(_mentioned(m, text) for m in spec.get("must_mention", []) or []),
        "avoids": not any(m.lower() in text for m in spec.get("must_not_mention", []) or []),
        "cites": (f.get("cited", False) or spec.get("expect_uncited_ok", False))
                 and all(any(c.startswith(p) for c in f.get("citations", [])) for p in spec.get("must_cite_prefix", []) or []),
        "in_budget": res["latency"]["total_ms"] <= budget_ms,
    }
    return {"q": spec["q"], "pass": all(checks.values()), "checks": checks, "latency_ms": res["latency"]["total_ms"],
            "answer": f.get("text", f.get("question", ""))[:300], "citations": f.get("citations", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--db", required=True)
    ap.add_argument("--episode", default=None, help="default: the latest episode in the store")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.scenario).read_text())
    engine = connect(a.db)
    ep = a.episode or sorted({e["episode_id"] for e in search_events(engine, limit=10000)})[-1]
    backend = {"fake": lambda: FakeBackend(), "openai": lambda: OpenAIBackend(base_url=a.base_url, model=a.model),
               "transformers": lambda: TransformersBackend(model_id=a.model)}[a.backend]()
    budget = int(spec.get("latency_budget_ms", 10000))
    results = [score(ask(engine, q["q"], ep, backend), q, budget) for q in spec["questions"]]
    unscored = spec.get("ground_truth", {}).get("people_total") is None
    row = {"ring": "scenario", "scenario": Path(a.scenario).name, "episode_id": ep, "backend": a.backend, "model": a.model,
           "passed": sum(r["pass"] for r in results), "total": len(results), "ground_truth_filled": not unscored,
           "latency_p50_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
           "latency_max_ms": max(r["latency_ms"] for r in results), "results": results,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "scenario.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    for r in results:
        flags = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["checks"].items())
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['latency_ms'] / 1000:5.1f}s  {r['q'][:70]}\n         {flags}\n         {r['answer'][:160]}")
    print(f"  {row['passed']}/{row['total']} passed, p50 {row['latency_p50_ms'] / 1000:.1f}s, max {row['latency_max_ms'] / 1000:.1f}s"
          + ("  (ground truth not filled in yet: mention rules are placeholders)" if unscored else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/scenarios/warehouse.yaml << 'EOF_VI'
# Acceptance scenario for the warehouse clip (HI_DEF_VIDEO.mp4, 17 s, 6 fps, 1280x720).
# Ground truth established by watching the clip (session 16b). bench/scenario_eval.py scores the
# agent against it; a mention rule given as a list means "any of these".
clip: HI_DEF_VIDEO.mp4
ground_truth:
  people_total: 8
  people_whole_time: 6
  people_at_table: 5
  people_right_side: 3
  people_at_conveyor: 2
  pickups: 0
  notes: |
    Table, whole clip: dark jacket + orange vest (near-left); orange vest, black hair (near-left);
    woman, orange vest over dark top (far-left end); white hard hat + orange vest + blue jeans
    (far side, moves right behind boxes ~6 s); pink/dark cap + orange vest (right of table at 0 s,
    far-middle by 8 s). Right side: yellow-green hi-vis + white hard hat at the conveyor 0-17 s;
    second hi-vis worker, dark cap, beside him at 0 s, walks along the conveyor ~8 s; dark jacket +
    orange vest walking boxes in the dock aisle top-right, first visible ~4 s, in and out after.
    Hanging jackets on the far-left racking are NOT people (false "person" detections there).
questions:
  - q: How many distinct people were in this episode, and which of them stayed the whole time?
    must_mention: [["8", "eight", "7", "seven", "9", "nine"]]     # 8, ±1 tolerance while ReID matures
    must_not_mention: ["14", "13", "12", "11"]
    must_cite_prefix: ["cam1:E"]
  - q: Who was at the conveyor on the right, and when did they arrive?
    must_mention: [["00:0", "start", "beginning", "0.0", "from the first", "already"]]   # the hard-hat worker is there from 0 s
    must_not_mention: []
    must_cite_prefix: ["cam1:E"]
    later: ["hard hat", "hi-vis"]       # becomes must_mention once the writer VLM supplies attributes
  - q: Did anyone pick something up from a shelf?
    must_mention: [["no", "none", "nothing", "did not", "didn't", "not"]]
    must_not_mention: ["yes, "]
    must_cite_prefix: []
    expect_uncited_ok: true
latency_budget_ms: 10000
EOF_VI
cat > tests/test_slice_gpu_smoke.py << 'EOF_VI'
import json
import os
import subprocess
import sys

import pytest
from pathlib import Path


@pytest.mark.parametrize("detect", ["roi", "frame", "hybrid"])
def test_slice_runs_end_to_end_with_fake_detector(tmp_path, detect):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", detect, "--fps", "5", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".",
                         env={**os.environ, "PYTHONPATH": os.getcwd()})   # this checkout, not another editable install
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["detect"] == detect and row["frames"] > 0 and row["episode_records"] > row["frames"]
    assert row["person_tubes"] >= 1
    if detect == "hybrid":
        assert row["heartbeats"] >= 1


def test_slice_runs_with_histogram_reid(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", "frame", "--fps", "5", "--reid", "hist", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["reid"] == "hist" and row["entities"] >= 1 and row["embed_ms_p50"] is not None


@pytest.mark.edge("E-DET-05")
def test_media_zone_suppresses_detections_and_zone_file_is_picked_up(tmp_path):
    pytest.importorskip("av")
    import json
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    zones_dir = tmp_path / "data" / "zones"; zones_dir.mkdir(parents=True)
    # a media zone covering the whole frame: every detection is suppressed -> no tubes at all
    (zones_dir / "walk.json").write_text(json.dumps([{"zone_id": "poster", "camera_id": "cam1", "kind": "media",
                                                      "polygon": [[0, 0], [320, 0], [320, 240], [0, 240]]}]))
    out = subprocess.run([sys.executable, str(Path.cwd() / "bench" / "slice_gpu.py"), "--source", str(clip), "--model", "fake",
                          "--detect", "frame", "--fps", "5", "--out", str(tmp_path / "data" / "episodes")],
                         capture_output=True, text=True, cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "zones: ['poster'] (file" in out.stdout
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["person_tubes"] == 0
EOF_VI
cat > edge_cases.yaml << 'EOF_VI'
# Edge-case registry. Source of truth for `make coverage`.
# status: implemented (code exists; MUST have a test marked @pytest.mark.edge("ID"))
#         planned     (in scope for month 1; coverage prints it as TODO)
#         deferred    (out of month-1 scope; listed so it is not forgotten)
# Adding a case: append here first, then code, then test. Never the other order.
version: 1
cases:
# ---------------- ingest / time ----------------
- id: E-ING-01
  ring: ingest
  title: Camera clocks disagree
  trigger: Cameras drift seconds apart; fusion hand-offs and the ±1 s fact-miner window fail.
  handling: Every timestamp is CamTime{cam_utc_ms, offset_ms, offset_confidence}; offsets estimated in the calibration walk and re-estimated from hand-offs; fusion uses corrected_ms() only.
  status: implemented
- id: E-ING-02
  ring: ingest
  title: Stream drops or reconnects
  trigger: RTSP disconnect mid-episode.
  handling: signal_lost/signal_restored events; open episode on that camera closes with status=truncated; tubes go to lost, not exited.
  status: implemented
- id: E-ING-03
  ring: ingest
  title: Variable or dropped frame rate
  trigger: Camera throttles under load; fps changes.
  handling: Ticks are wall-clock windows (t_start,t_end), never frame indices; tube age computed from corrected time.
  status: implemented
- id: E-ING-04
  ring: ingest
  title: Codec without motion vectors
  trigger: MJPEG or unknown codec stream.
  handling: Gate interface is codec-agnostic; FrameDiffGate fallback at 1 fps; camera profile records gate_mode.
  status: implemented
- id: E-ING-05
  ring: ingest
  title: Resolution or aspect change mid-stream
  trigger: Camera reconfigured; homography and zones now invalid.
  handling: Reader SizeGuard flags the change; EventCompiler.on_size_change emits camera_moved_suspect; fusion freezes that camera until recalibration (week 3).
  status: implemented
- id: E-ING-06
  ring: ingest
  title: Timezone and DST
  trigger: '"Yesterday afternoon" across a DST switch or a site in another timezone.'
  handling: All storage in UTC ms; site timezone stored in KB; agent resolves local windows explicitly and echoes them in the answer.
  status: implemented
- id: E-ING-07
  ring: ingest
  title: Duplicate or out-of-order frames
  trigger: RTSP jitter delivers repeated or reordered PTS.
  handling: Dedupe by PTS per camera; drop frames older than the last processed tick.
  status: implemented
# ---------------- ring 0 gate ----------------
- id: E-GATE-01
  ring: ring0
  title: Stationary object is invisible to the gate
  trigger: Person stops moving; bag placed and left; fallen person still.
  handling: HeartbeatScheduler runs a full-frame detection at episode open, every 1 s on active tiles and every 10 s on quiet ones (merged with ROI detections); measured on real footage as person_tubes and lost-vs-exited in bench/slice_gpu.py --detect hybrid.
  status: implemented
- id: E-GATE-02
  ring: ring0
  title: Lighting change looks like whole-frame motion
  trigger: Light switch, IR-cut toggle, cloud passes.
  handling: Global luminance step detected before differencing; emitted as illumination_change scene-state event; blobs suppressed for that update.
  status: implemented
- id: E-GATE-03
  ring: ring0
  title: Foliage, rain, insects, sensor noise
  trigger: Persistent low-level motion that is not an object.
  handling: Adaptive noise floor (median block energy over history) × k; MV-field coherence test once MVGate lands.
  status: implemented
- id: E-GATE-04
  ring: ring0
  title: Camera shake or PTZ move
  trigger: Most of the frame changes at once.
  handling: active_fraction > threshold => global_motion; blobs suppressed, background frozen; sustained => camera_moved_suspect.
  status: implemented
- id: E-GATE-05
  ring: ring0
  title: Very slow motion under threshold
  trigger: Someone creeping; object slid slowly.
  handling: Same HeartbeatScheduler full-frame pass catches motion below the gate threshold; gate FN rate stays a tracked metric.
  status: implemented
- id: E-GATE-06
  ring: ring0
  title: Motion in screens, mirrors, reflections
  trigger: TV playing; mirror shows a person in another zone; steel reflects motion.
  handling: Scene card media_zones and reflective_surfaces become gate masks; tubes born inside them are reflection_suspect.
  status: planned
- id: E-GATE-07
  ring: ring0
  title: Intra-only or all-I-frame streams
  trigger: Encoder configured with no P-frames; no motion vectors.
  handling: Same fallback as E-ING-04.
  status: planned
- id: E-GATE-08
  ring: ring0
  title: Night noise inflates motion energy
  trigger: High-gain low-light mode.
  handling: Per-modality threshold profile; coherence weighted higher at night.
  status: planned
# ---------------- ring 1 detect ----------------
- id: E-DET-01
  ring: ring1
  title: Tiny objects below detector floor
  trigger: Keys, phone, small tools at room-camera resolution.
  handling: Not detected as tubes; tracked via carried_item attribute and asset-home occupancy; inspect tool for on-demand native-res count/find; resolution ceiling stated in answers.
  status: planned
- id: E-DET-02
  ring: ring1
  title: Truncation at frame edge
  trigger: Half a person at the border.
  handling: remap_detections flags boxes within 2 px of the frame border as truncated; Ring 2 then uses fallback_bbox_bottom with wide uncertainty and keyframe scoring penalises them.
  status: implemented
- id: E-DET-03
  ring: ring1
  title: Class confusion
  trigger: Child vs small adult; dog vs bag; mannequin vs person.
  handling: Class is an attribute with confidence; enrichment can override; agent never asserts class below threshold.
  status: planned
- id: E-DET-04
  ring: ring1
  title: Ghost detections in reflections
  trigger: Mirror/glass shows a duplicate person.
  handling: Tubes inside reflective_surfaces are reflection_suspect; fusion never creates an entity from a suspect tube alone.
  status: planned
- id: E-DET-05
  ring: ring1
  title: People on screens, posters, photos
  trigger: TV shows a face; framed photo detected as person.
  handling: zones of kind `media` (scene card, or data/zones/<clip>.json) drop any detection whose foot point falls inside them; the warehouse clip's hanging jackets on the left racking are the first case.
  status: implemented
- id: E-DET-06
  ring: ring1
  title: Dense crowds
  trigger: Heavy overlap; fragmentation.
  handling: crowd event on density; tube quality flag; enrichment skipped for low-quality tubes.
  status: deferred
- id: E-DET-07
  ring: ring1
  title: Fisheye or wide-lens distortion
  trigger: Homography from a distorted image.
  handling: fov_class from scene card; undistort before foot-point projection.
  status: deferred
- id: E-DET-08
  ring: ring1
  title: IR appearance shift lowers recall
  trigger: Night mode.
  handling: Measured on night eval set before any enhancer; enhancer only if gap is real and license clear.
  status: planned
- id: E-DET-09
  ring: ring1
  title: Static furniture and fixtures become tubes
  trigger: Detector emits dining table, tv, chair as objects; tracker births permanent tubes for them.
  handling: Only TUBE_CLASSES (people, animals, vehicles, carried bags) become tubes; furniture is a scene-card asset owned by zones and heartbeats.
  status: implemented
- id: E-DET-10
  ring: ring1
  title: One object, several detections
  trigger: A person straddles two adjacent ROI crops, or is seen by both a crop and the full-frame heartbeat with boxes of different extent; each becomes a tube.
  handling: Every tick's detections pass through dedupe_detections (class-wise, IoU>=0.5 or intersection-over-smaller>=0.6); complete boxes beat crop-truncated ones. Measured in session 06 as max_concurrent 13 on a 9-person frame.
  status: implemented
- id: E-DET-11
  ring: ring1
  title: Zoomed crops promote background specks and boxes into confident detections
  trigger: In an ROI crop a 20-px figure at the far dock reads as person 0.6 and cardboard as suitcase 0.5; each becomes a tube (13 suitcase tubes in session 05).
  handling: 'Birth gates in the tracker: minimum box height (32 px) and class-specific confidence (person 0.5, other classes 0.65); re-attachment thresholds unchanged.'
  status: implemented
# ---------------- ring 2 tubes / fusion ----------------
- id: E-TUBE-01
  ring: ring2
  title: ID switch when paths cross
  trigger: Two people cross; IoU ambiguous.
  handling: Keep best match; record other as merge_candidate; never silently merge; fusion may later resolve with ReID.
  status: implemented
- id: E-TUBE-02
  ring: ring2
  title: Occlusion
  trigger: Person behind furniture or another person.
  handling: state=occluded with occluded_since; after max_occluded_ms becomes lost (or exited if last box touched an exit zone).
  status: implemented
- id: E-TUBE-03
  ring: ring2
  title: Long stationary dwell
  trigger: Sleeping person; parked object.
  handling: Heartbeat detections with det_source=heartbeat keep the track active.
  status: implemented
- id: E-TUBE-04
  ring: ring2
  title: Re-entry after leaving
  trigger: Same person returns minutes later.
  handling: TubeLinker ties a newborn tube to an entity whose tube went lost within 30 s and 400 px when appearance cosine >= 0.75; emits Event(relink); the agent reports entities, not tubes.
  status: implemented
- id: E-TUBE-05
  ring: ring2
  title: Same person in two overlapping cameras
  trigger: Overlap region.
  handling: Per-camera tracker never merges; fusion merges by floor distance + ReID + time; enrichment once per entity, best view elected.
  status: planned
- id: E-TUBE-06
  ring: ring2
  title: Hand-off outside transit bounds
  trigger: Entity appears in a non-adjacent tile too fast.
  handling: impossible_transition event; no merge; anomaly surfaced.
  status: planned
- id: E-TUBE-07
  ring: ring2
  title: Blind-spot ambiguity
  trigger: Two enter a blind spot, one exits.
  handling: Delay-state keeps both candidates with probabilities; answer states ambiguity.
  status: deferred
- id: E-TUBE-08
  ring: ring2
  title: Appearance drift within a day
  trigger: Jacket on/off; bag picked up.
  handling: Per-entity EMA embedding plus up to 5 exemplars refreshed on active ticks; matching takes the best of EMA and exemplars so a jacket-off entity still links.
  status: implemented
- id: E-TUBE-09
  ring: ring2
  title: Carried object is not its own tube
  trigger: Bag on shoulder; keys in hand.
  handling: carried_item attribute on the person tube; object tube only when placed and stationary.
  status: planned
- id: E-TUBE-10
  ring: ring2
  title: Pets, strollers, wheelchairs
  trigger: Non-person moving classes; child inside stroller.
  handling: Class-specific lifecycle; stroller+child handled as one tube with carried_item=child hint.
  status: deferred
- id: E-TUBE-13
  ring: ring2
  title: Fragmentation at sampled frame rate
  trigger: Decode at 2 fps means a walking person moves half a box width between ticks; IoU association breaks the tube.
  handling: ByteTracker with dt-aware Kalman, buffered IoU and a first-tick centre gate keeps fragmentation at 1.0 down to ~4-6 fps (synthetic); below ~3 fps motion-only association is ambiguous by construction, so active tiles decode at >=4 fps and 2 fps is for quiet tiles only.
  status: implemented
- id: E-TUBE-14
  ring: ring2
  title: Predicted box balloons during long occlusion
  trigger: Kalman aspect/height velocities keep integrating while a track is unseen; the predicted box grows over the scene and re-attaches to junk (seen in the session-08 debug frames).
  handling: KalmanBoxFilter clamps predicted height and aspect to 0.6–1.6× the last measurement and zeroes the size velocities when the clamp engages.
  status: implemented
- id: E-TUBE-12
  ring: ring2
  title: Feet occluded, foot point wrong
  trigger: Person behind counter.
  handling: FloorPoint.source=fallback_bbox_bottom and uncertainty widened; fusion tolerances read it.
  status: implemented
# ---------------- ring 3a events ----------------
- id: E-EVT-01
  ring: ring3a
  title: Zone boundary jitter
  trigger: Foot point oscillates on a zone edge.
  handling: 'Hysteresis: enter after N consecutive inside ticks, exit after N outside.'
  status: implemented
- id: E-EVT-10
  ring: ring3a
  title: Everyone in frame at episode open "enters"
  trigger: First tick births every visible tube inside a zone; enter_zone fires for all of them at once.
  handling: EventCompiler seeds zone memberships silently for open_grace_ms (1.5 s) after its first tick, since the gate needs a frame to warm up; the open heartbeat makes that first tick non-empty.
  status: implemented
- id: E-EVT-02
  ring: ring3a
  title: Cross-camera event order under clock offset
  trigger: Kitchen camera 2 s ahead of hallway.
  handling: Sort by corrected_ms; hand-off window tolerates offset_confidence.
  status: planned
- id: E-EVT-03
  ring: ring3a
  title: False pickup from occlusion of the shelf
  trigger: Person stands in front of the asset home; heartbeat cannot see the asset.
  handling: Pickup requires asset absent on a heartbeat taken with nobody inside the zone, after a present reading; subject = visitors in between; no visitors => asset_missing_from_home with no blame.
  status: implemented
- id: E-EVT-04
  ring: ring3a
  title: Fall vs lying down on purpose
  trigger: Person lies on sofa/bed.
  handling: rest zones from scene card suppress fall; fall requires vertical velocity + pose primitive outside rest zones.
  status: deferred
- id: E-EVT-05
  ring: ring3a
  title: Left-behind vs placed at home
  trigger: Bag set on its usual hook.
  handling: left_behind only outside the object's home zone.
  status: planned
- id: E-EVT-06
  ring: ring3a
  title: Duplicate events from overlapping cameras
  trigger: Both cameras see the same enter.
  handling: dedupe_key = type|entity|time bucket at fusion; per-camera events keep camera_id for evidence.
  status: implemented
- id: E-EVT-07
  ring: ring3a
  title: Episode never closes
  trigger: Busy lobby active for hours.
  handling: soft_cut on cast churn > threshold or max duration; events may span episodes.
  status: implemented
- id: E-EVT-08
  ring: ring3a
  title: Event spans an episode boundary
  trigger: Dwell starts in one episode and ends in the next.
  handling: Events carry episode_id of emission; queries join across episodes by entity and time, never by episode alone.
  status: planned
# ---------------- ring 3b foveation / enrichment ----------------
- id: E-FOV-01
  ring: ring3b
  title: Crop too small to describe
  trigger: Short side < 96 px.
  handling: Enhancement ladder; enhanced=true propagates to attributes; confidence discounted.
  status: implemented
- id: E-FOV-02
  ring: ring3b
  title: Motion blur
  trigger: Fast movement.
  handling: Keyframe score uses Laplacian sharpness; blurred frames skipped; if none sharp, attributes low confidence.
  status: planned
- id: E-FOV-03
  ring: ring3b
  title: Backlit or overexposed subject
  trigger: Doorway against daylight.
  handling: Exposure score in keyframe selection; try other keyframes; color_reason=low_confidence.
  status: planned
- id: E-FOV-04
  ring: ring3b
  title: Cross-cell attribute bleed in contact sheets
  trigger: VLM describes cell 3 with cell 4's jacket.
  handling: Hard borders + cell ids; fixed expected_cells; per-cell schema; bleed rate measured on eval set; fall back to 12 cells if >5%.
  status: implemented
- id: E-FOV-05
  ring: ring3b
  title: Hallucinated color in IR
  trigger: Monochrome crop.
  handling: Attributes validator and grammar forbid colors when modality is ir/thermal; color_reason=ir_mode; tone fields instead.
  status: implemented
- id: E-FOV-06
  ring: ring3b
  title: Ring buffer expired before enrichment
  trigger: Slow path lags past buffer window.
  handling: Best keyframe crop persisted to R2 at tube birth; keyframe_refs on the tube.
  status: implemented
- id: E-FOV-07
  ring: ring3b
  title: VLM output violates schema
  trigger: Grammar bug or truncation.
  handling: Retry once; then EnrichmentPatch{failed=true}; tick never blocked.
  status: implemented
- id: E-FOV-08
  ring: ring3b
  title: Attributes contradict across keyframes
  trigger: Red jacket in one, dark in another.
  handling: Majority vote over keyframes; keep history; confidence reflects agreement.
  status: planned
# ---------------- storage / handoff ----------------
- id: E-STO-01
  ring: storage
  title: Patch arrives after episode closed
  trigger: Slow path finishes late.
  handling: Episode file is append-only; PatchRecord accepted after close.
  status: implemented
- id: E-STO-02
  ring: storage
  title: Retries create duplicates
  trigger: Writer crashes and replays.
  handling: Deterministic episode/event ids; per-record idempotency keys.
  status: implemented
- id: E-STO-03
  ring: storage
  title: Retention expired for a clip
  trigger: Clip requested for footage past retention.
  handling: clip tool checks availability; degrades to keyframes; answer states retention.
  status: planned
- id: E-STO-04
  ring: storage
  title: Schema version bump
  trigger: New field added mid-deployment.
  handling: Provenance.schema_version on every record; readers tolerate unknown fields; migration script per bump.
  status: implemented
- id: E-STO-05
  ring: storage
  title: KB version drift
  trigger: Transit bounds changed after an episode was compiled.
  handling: Provenance.kb_version on every record; replays pin the version.
  status: implemented
# ---------------- block 2 agent ----------------
- id: E-AGT-01
  ring: agent
  title: Ambiguous anchor
  trigger: '"After Jay went out" matches two exits.'
  handling: One clarifying question; never a silent pick.
  status: planned
- id: E-AGT-02
  ring: agent
  title: No results
  trigger: Nothing matches the window.
  handling: Empty tool results are shown to the model as such; the answer must say nothing matched and suggest widening; citations empty.
  status: implemented
- id: E-AGT-03
  ring: agent
  title: Unknown named entity
  trigger: '"Jay" not in gallery.'
  handling: Ask who Jay is; offer naming form.
  status: planned
- id: E-AGT-04
  ring: agent
  title: Fuzzy time expressions and guessed vocabulary
  trigger: '"yesterday afternoon", "after lunch"; the model invents event names like pick_up.'
  handling: Event and zone vocabularies are in the system prompt; tool args are validated and a miss returns the valid list so the model corrects itself; time windows are echoed in answers.
  status: implemented
- id: E-AGT-05
  ring: agent
  title: Color question on IR footage
  trigger: '"what color was his shirt" at night.'
  handling: State IR mode; give tone; never a color.
  status: planned
- id: E-AGT-06
  ring: agent
  title: Unverified claim
  trigger: Attribute from a low-confidence patch.
  handling: The agent loop keeps the set of ids that appeared in tool results; an answer whose citations are not in that set is sent back once for revision, then returned with cited=false. verify() lands with the writer VLM.
  status: implemented
- id: E-AGT-07
  ring: agent
  title: Broken custody chain
  trigger: carried_item missed; drop never fired.
  handling: 'Graceful degrade: last known custody + offer clip.'
  status: planned
- id: E-AGT-08
  ring: agent
  title: Duplicates from overlapping cameras in the answer
  trigger: Same person listed twice.
  handling: Answer over entities, not tubes.
  status: planned
# ---------------- KB / calibration ----------------
- id: E-KB-01
  ring: kb
  title: Camera moved after calibration
  trigger: Bumped or remounted.
  handling: Sustained global motion => camera_moved_suspect; landmark check on I-frames; freeze fusion; prompt recalibration.
  status: implemented
- id: E-KB-02
  ring: kb
  title: Scene card labels wrong
  trigger: '"Dining hall" is actually the kitchen.'
  handling: Everything on the card is a hypothesis; user correction bumps version and re-labels.
  status: planned
- id: E-KB-03
  ring: kb
  title: Causal fact contradicted
  trigger: Switch flipped, light did not change.
  handling: Fact.add_contradiction; retire at 2 unless user_confirmed.
  status: implemented
- id: E-KB-04
  ring: kb
  title: Furniture moved
  trigger: Shelf relocated; asset home zone stale.
  handling: Repeated asset_missing with asset detected elsewhere => propose new home.
  status: deferred
# ---------------- night ----------------
- id: E-NIGHT-01
  ring: night
  title: IR switch mid-tube
  trigger: Camera toggles IR while a person is in view.
  handling: modality_switch event; tube segments carry modality; attributes per segment.
  status: planned
- id: E-NIGHT-02
  ring: night
  title: ReID does not transfer day to IR
  trigger: Same person, different modality.
  handling: Separate exemplar sets per modality; tile/time continuity weighted higher at night.
  status: planned
- id: E-NIGHT-03
  ring: night
  title: IR blooming near camera
  trigger: Subject washed out by illuminator.
  handling: Exposure score rejects keyframe; low confidence attributes.
  status: deferred
# ---------------- privacy ----------------
- id: E-PRIV-01
  ring: privacy
  title: Gallery entry deletion
  trigger: User deletes a named person.
  handling: 'Cascade: exemplars, names on records, KB facts referencing the entity; anonymous ids remain.'
  status: planned
- id: E-PRIV-02
  ring: privacy
  title: Face embeddings without opt-in
  trigger: Face model enabled by default.
  handling: Site-level opt-in flag gates the face lane; default off.
  status: planned
EOF_VI
chmod +x colab/vllm_venv.sh colab/preflight.sh 2>/dev/null || true
cp "$0" colab/sessions/session_18_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. preflight"
bash colab/preflight.sh | tee data/bench/preflight.json

step "4. install + harness"
pipi -e ".[dev,ingest]"
make check

step "5. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"
  fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" 2>/dev/null; sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"
    DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "6. slice with real zones (frame + SigLIP) -> episode -> store"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "^zones:|\"(person_tubes|person_tubes_low_quality|entities|relinks|max_concurrent_persons)\"" /tmp/slice_final.log | sed 's/^/  /'
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "7. reasoning backend: vLLM (clean env, runtime-matched torch) -> transformers -> fake"
BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if [ "$HAS_GPU" = 1 ] && [ "${SKIP_VLLM:-0}" != "1" ]; then
    if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai; else warn "vLLM path failed -> transformers in-process"; fi
  fi
  if [ -z "$BACKEND" ] && [ "$HAS_GPU" = 1 ]; then python -c "import transformers" 2>/dev/null || pipi transformers; BACKEND=transformers; fi
  [ -z "$BACKEND" ] && { BACKEND=fake; warn "no GPU: fake backend"; }
fi
echo "backend: $BACKEND ($AGENT_MODEL)"

step "8. scene script head + the three questions with latency"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 8 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]" | grep -v "HF Hub"

step "9. scenario acceptance (8 people, 6 whole time, 0 pickups, 10 s budget)"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -v "HF Hub\|Loading weights" | tail -12

step "10. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: clean-env vLLM launch matched to runtime torch, real zones for the warehouse clip, media-zone suppression (E-DET-05), id-safe scenario rules"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "11. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 6, 7, 8 and 9 back."
