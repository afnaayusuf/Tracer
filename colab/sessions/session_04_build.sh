#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 04 build: audit the repo, add the real-footage slice,
#  keyframe store, MOT evaluator and Ring 2 baseline bench; run everything; commit; push.
#
#  Colab: paste this whole file into one %%bash cell (after the env cell that sets
#  GH_TOKEN and GH_REPO), or upload it and run:  bash /content/build_session_04.sh
#
#  Env (all optional):
#    GH_REPO   owner/repo            default afnaayusuf/Tracer
#    GH_TOKEN  fine-grained PAT      needed only to push
#    REPO_DIR  checkout path         default /content/Tracer
#    SOURCE    clip for the slice    default /content/HI_DEF_VIDEO.mp4
#    NO_PUSH=1 skip git push         FORCE=1 continue past audit warnings
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 04"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q origin
  git -C "$REPO_DIR" pull -q --ff-only || die "local branch diverged from origin; resolve manually"
else
  git clone -q "https://github.com/$GH_REPO.git" "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -n "${GH_TOKEN:-}" ]; then
  git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
fi
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present; commit or discard them first: $(git status --short | tr '\n' ' ')"
[ -n "$(git log --grep='^session 03:' --format=%h)" ] || die "session 03 commit not found in history; this script expects the session-03 tree"
if grep -rn "import cv2\|from cv2\|opencv" vi bench tests pyproject.toml 2>/dev/null; then
  [ "${FORCE:-0}" = "1" ] || die "cv2/opencv found in code (bypasses vi.ingest); remove it or rerun with FORCE=1"
fi
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/sessions/session_02_apply.sh
colab/sessions/session_03_apply.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_schemas.py
tests/test_tubes.py
vi/__init__.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
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
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
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
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/geometry.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ -f test.txt ]; then git rm -q test.txt; echo "removed stray test.txt"; extras=$((extras-1)); fi
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  die "$extras unexpected file(s); inspect them, delete or commit deliberately, or rerun with FORCE=1"
fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p vi/eval colab/sessions data/bench
cat > vi/detect/base.py << 'EOF_VI'
from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from vi.schemas import Box


# Classes that become tubes. Furniture, fixtures and appliances are scene-card assets handled by
# heartbeats and zones, never tracked as moving objects (E-DET-09).
TUBE_CLASSES = {"person", "dog", "cat", "bicycle", "car", "motorcycle", "bus", "truck",
                "backpack", "handbag", "suitcase", "umbrella"}


def is_tube_class(label: str) -> bool:
    return label in TUBE_CLASSES


class Detection(BaseModel):
    box: Box
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = None   # R14: ReID alongside the box
    truncated: bool = False                # E-DET-02: box touches the frame border


class Detector(Protocol):
    """Ring 1 interface. Batched over ROI crops from many cameras (R12)."""

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]: ...
EOF_VI
cat > vi/detect/__init__.py << 'EOF_VI'
from .base import TUBE_CLASSES, Detection, Detector, is_tube_class
from .roi import ROI, blobs_to_rois, crop_roi, pad_batch, remap_detections
EOF_VI
cat > vi/events/zones.py << 'EOF_VI'
from __future__ import annotations

from pydantic import BaseModel, Field


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


class Zone(BaseModel):
    """Image-space polygon for one camera (world-space zones come with fusion)."""

    zone_id: str
    camera_id: str
    tile_id: str | None = None
    polygon: list[tuple[float, float]] = Field(min_length=3)
    kind: str = "generic"           # generic | exit | asset_home | actuator | media | rest
    asset_id: str | None = None     # for asset_home zones

    def contains(self, pt: tuple[float, float]) -> bool:
        return point_in_polygon(pt, self.polygon)


def default_zones(camera_id: str, width: int, height: int, edge_frac: float = 0.1,
                  tile_id: str | None = None) -> list[Zone]:
    """Two exit strips (left/right edges) and a centre floor zone, in native pixels. Enough
    for the first real-footage slice; real zones come from the scene card + walk."""
    e = width * edge_frac
    return [
        Zone(zone_id="exit_left", camera_id=camera_id, tile_id=tile_id, kind="exit",
             polygon=[(0, 0), (e, 0), (e, height), (0, height)]),
        Zone(zone_id="exit_right", camera_id=camera_id, tile_id=tile_id, kind="exit",
             polygon=[(width - e, 0), (width, 0), (width, height), (width - e, height)]),
        Zone(zone_id="floor_centre", camera_id=camera_id, tile_id=tile_id, kind="generic",
             polygon=[(e, height * 0.4), (width - e, height * 0.4), (width - e, height), (e, height)]),
    ]


def load_zones(path: str, camera_id: str | None = None) -> list[Zone]:
    """JSON: [{"zone_id":..., "camera_id":..., "kind":..., "polygon":[[x,y],...], "asset_id":...}, ...]"""
    import json
    from pathlib import Path

    items = json.loads(Path(path).read_text())
    zones = [Zone(**{**z, "polygon": [tuple(p) for p in z["polygon"]]}) for z in items]
    return [z for z in zones if camera_id is None or z.camera_id == camera_id]
EOF_VI
cat > vi/events/__init__.py << 'EOF_VI'
from .zones import Zone, default_zones, load_zones, point_in_polygon
from .compiler import EventCompiler
EOF_VI
cat > vi/episode/keyframes.py << 'EOF_VI'
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from vi.schemas import Box


class KeyframeStore:
    """E-FOV-06 / R20: persist a padded native-resolution crop the moment a tube is born,
    before the ring buffer can expire. Local JPEGs now; the same refs point at R2 later."""

    def __init__(self, root: str | Path, pad: float = 0.2, quality: int = 90):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pad = pad
        self.quality = quality
        self.saved = 0

    def save(self, camera_id: str, t_ms: int, box: Box, frame_rgb: np.ndarray, tag: str = "birth") -> str:
        h, w = frame_rgb.shape[:2]
        px, py = box.width * self.pad, box.height * self.pad
        x1, y1 = int(max(0, box.x1 - px)), int(max(0, box.y1 - py))
        x2, y2 = int(min(w, box.x2 + px)), int(min(h, box.y2 + py))
        crop = frame_rgb[y1:y2, x1:x2]
        rel = Path(camera_id) / f"{t_ms}_{tag}_{x1}_{y1}.jpg"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(np.ascontiguousarray(crop)).save(path, quality=self.quality)
        except ImportError:  # keep going without Pillow: raw .npy
            path = path.with_suffix(".npy")
            np.save(path, crop)
        self.saved += 1
        return f"kf://{rel.with_suffix(path.suffix).as_posix()}"

    def make_sink(self, current_frame: Callable[[], np.ndarray | None]) -> Callable[[str, int, Box], str]:
        """Adapter for SimpleIoUTracker(keyframe_sink=...): the tracker only knows the box,
        the closure knows the frame."""

        def sink(camera_id: str, t_ms: int, box: Box) -> str:
            frame = current_frame()
            if frame is None:
                return f"kf://missing/{camera_id}/{t_ms}"
            return self.save(camera_id, t_ms, box, frame)

        return sink
EOF_VI
cat > vi/episode/__init__.py << 'EOF_VI'
from .keyframes import KeyframeStore
from .writer import EpisodeWriter, episode_id_for, should_soft_cut
EOF_VI
cat > vi/eval/__init__.py << 'EOF_VI'
from .mot import MOTResult, evaluate_mot, load_mot_txt
EOF_VI
cat > vi/eval/mot.py << 'EOF_VI'
"""CLEAR-MOT (MOTA/MOTP/IDSW) and IDF1 from first principles, so Ring 2 has a baseline
number before any external evaluator is installed. Inputs are {frame: [(id, Box), ...]}.
Matching follows the CLEAR convention: keep last frame's pairing while IoU stays above the
threshold, Hungarian-match the rest."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from vi.schemas import Box

Tracks = dict[int, list[tuple[int, Box]]]


@dataclass
class MOTResult:
    num_gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    iou_sum: float = 0.0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    gt_ids: int = 0
    pred_ids: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mota(self) -> float:
        return 1.0 - (self.fn + self.fp + self.idsw) / self.num_gt if self.num_gt else 0.0

    @property
    def motp(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def idf1(self) -> float:
        d = 2 * self.idtp + self.idfp + self.idfn
        return 2 * self.idtp / d if d else 0.0

    @property
    def fragmentation_ratio(self) -> float:
        """pred ids per gt id: 1.0 is perfect; 3.0 means every person became three tubes."""
        return self.pred_ids / self.gt_ids if self.gt_ids else 0.0

    def as_row(self) -> dict:
        return {"mota": round(self.mota, 4), "motp": round(self.motp, 4), "idf1": round(self.idf1, 4),
                "idsw": self.idsw, "fp": self.fp, "fn": self.fn, "tp": self.tp, "num_gt": self.num_gt,
                "gt_ids": self.gt_ids, "pred_ids": self.pred_ids,
                "fragmentation_ratio": round(self.fragmentation_ratio, 3), **self.extra}


def _iou_matrix(a: list[Box], b: list[Box]) -> np.ndarray:
    m = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            m[i, j] = x.iou(y)
    return m


def evaluate_mot(gt: Tracks, pred: Tracks, iou_thr: float = 0.5) -> MOTResult:
    res = MOTResult()
    frames = sorted(set(gt) | set(pred))
    last_match: dict[int, int] = {}                 # gt id -> pred id from previous frame
    overlap: dict[tuple[int, int], int] = {}        # (gt id, pred id) -> co-occurrence frames
    gt_frames: dict[int, int] = {}
    pred_frames: dict[int, int] = {}
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        res.num_gt += len(g)
        for gid, _ in g:
            gt_frames[gid] = gt_frames.get(gid, 0) + 1
        for pid, _ in p:
            pred_frames[pid] = pred_frames.get(pid, 0) + 1
        if not g or not p:
            res.fn += len(g)
            res.fp += len(p)
            continue
        iou = _iou_matrix([b for _, b in g], [b for _, b in p])
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        pairs: list[tuple[int, int]] = []
        # 1. keep previous pairings that still overlap
        pid_index = {pid: j for j, (pid, _) in enumerate(p)}
        for i, (gid, _) in enumerate(g):
            pid = last_match.get(gid)
            if pid is not None and pid in pid_index and iou[i, pid_index[pid]] >= iou_thr:
                pairs.append((i, pid_index[pid]))
                matched_g.add(i)
                matched_p.add(pid_index[pid])
        # 2. Hungarian on the rest
        gi = [i for i in range(len(g)) if i not in matched_g]
        pj = [j for j in range(len(p)) if j not in matched_p]
        if gi and pj:
            cost = 1.0 - iou[np.ix_(gi, pj)]
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if iou[gi[a], pj[b]] >= iou_thr:
                    pairs.append((gi[a], pj[b]))
                    matched_g.add(gi[a])
                    matched_p.add(pj[b])
        for i, j in pairs:
            gid, pid = g[i][0], p[j][0]
            res.tp += 1
            res.iou_sum += iou[i, j]
            if gid in last_match and last_match[gid] != pid:
                res.idsw += 1
            last_match[gid] = pid
            overlap[(gid, pid)] = overlap.get((gid, pid), 0) + 1
        res.fn += len(g) - len(pairs)
        res.fp += len(p) - len(pairs)
    # IDF1: global one-to-one assignment of gt ids to pred ids maximising co-occurrence
    gids = sorted(gt_frames)
    pids = sorted(pred_frames)
    res.gt_ids, res.pred_ids = len(gids), len(pids)
    total_gt = sum(gt_frames.values())
    total_pred = sum(pred_frames.values())
    if gids and pids:
        m = np.zeros((len(gids), len(pids)))
        for (gid, pid), n in overlap.items():
            m[gids.index(gid), pids.index(pid)] = n
        r, c = linear_sum_assignment(-m)
        res.idtp = int(m[r, c].sum())
    res.idfp = total_pred - res.idtp
    res.idfn = total_gt - res.idtp
    return res


def load_mot_txt(path: str | Path, gt: bool = False, min_conf: float = 0.0) -> Tracks:
    """MOTChallenge text: frame,id,x,y,w,h,conf,class,visibility. For gt.txt keep
    pedestrians (class 1) flagged as considered (conf==1)."""
    tracks: Tracks = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split(",")[:9]]
        frame, tid, x, y, w, h = int(v[0]), int(v[1]), v[2], v[3], v[4], v[5]
        conf = v[6] if len(v) > 6 else 1.0
        cls = int(v[7]) if len(v) > 7 else 1
        if gt and (conf != 1 or cls != 1):
            continue
        if not gt and conf < min_conf:
            continue
        if w <= 0 or h <= 0:
            continue
        tracks.setdefault(frame, []).append((tid, Box(x1=x, y1=y, x2=x + w, y2=y + h)))
    return tracks
EOF_VI
cat > bench/slice_gpu.py << 'EOF_VI'
"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source /content/HI_DEF_VIDEO.mp4 --fps 2 --model nano
  python bench/slice_gpu.py --source clip.mp4 --zones data/zones/cam1.json --tile lobby
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi.detect import blobs_to_rois, crop_roi, is_tube_class, pad_batch, remap_detections
from vi.detect.rfdetr import RFDETRDetector
from vi.episode import EpisodeWriter, KeyframeStore
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.tubes import SimpleIoUTracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--tile", default="tile1")
    ap.add_argument("--model", default="nano")
    ap.add_argument("--fps", type=float, default=4.0, help="R11 range 2-5; IoU-only tracking collapses below ~4")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default = edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=a.batch)
    gate = FrameDiffGate(a.camera)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    current = {"frame": None}
    zones = None
    tracker = compiler = writer = ep = None
    stage = Counter()
    ev_types: Counter = Counter()
    births = 0
    closed_all = []
    tick_ms = int(1000 / a.fps)
    frames = 0

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones = load_zones(a.zones, a.camera) if a.zones else default_zones(a.camera, w, h, tile_id=a.tile)
            exit_boxes = [z.polygon for z in zones if z.kind == "exit"]
            from vi.schemas import Box
            exits = [Box(x1=min(p[0] for p in poly), y1=min(p[1] for p in poly),
                         x2=max(p[0] for p in poly), y2=max(p[1] for p in poly)) for poly in exit_boxes]
            tracker = SimpleIoUTracker(a.camera, iou_thr=a.iou_thr, max_occluded_ms=a.max_occluded_ms,
                                       exit_boxes=exits, keyframe_sink=kf.make_sink(lambda: current["frame"]))
            compiler = EventCompiler(a.camera, zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=a.tile)
            writer = EpisodeWriter(a.out)
            ep = writer.open(a.tile, [a.camera], CamTime(cam_utc_ms=fr.pts_ms), prov)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        stage["gate"] += time.perf_counter() - t0
        events = compiler.on_gate(g)
        t0 = time.perf_counter()
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride)
        dets = []
        for i in range(0, len(rois), a.batch):
            chunk = rois[i:i + a.batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], a.batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = [d for d in dets if is_tube_class(d.class_label)]
        stage["detect"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms)
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
    writer.close(ep, CamTime(cam_utc_ms=fr.pts_ms + tick_ms), EpisodeStatus.closed, cast)
    st = reader.stats
    row = {
        "ring": "slice", "source": Path(a.source).name, "model": f"rf-detr-{a.model}", "sampled_fps": a.fps,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items()},
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
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
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}")


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/ring2_tubes.py << 'EOF_VI'
"""Ring 2 bench: tracker vs ground truth -> MOTA / IDF1 / IDSW / fragmentation. Baseline for the
ByteTrack rewrite. Rows to data/bench/ring2.jsonl.

  python bench/ring2_tubes.py --synthetic
  python bench/ring2_tubes.py --mot17 /content/drive/MyDrive/MOT17/train/MOT17-02-FRCNN --fps 30
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi.detect import Detection
from vi.eval import evaluate_mot, load_mot_txt
from vi.schemas import Box
from vi.tubes import SimpleIoUTracker


def synthetic_gt(frames: int = 120, seed: int = 0):
    """Two walkers crossing paths plus one that stops for 20 frames (stationary + crossing cases)."""
    rng = np.random.default_rng(seed)
    gt: dict[int, list[tuple[int, Box]]] = {}
    for f in range(frames):
        xa = 20 + f * 5
        xb = 620 - f * 5
        xc = 300 if 40 <= f < 60 else 300 + (f - 60) * 3 if f >= 60 else 300 - (40 - f) * 3
        items = []
        for tid, x in ((1, xa), (2, xb), (3, xc)):
            j = rng.normal(0, 1.5, 4)
            items.append((tid, Box(x1=x + j[0], y1=100 + j[1], x2=x + 40 + j[2], y2=220 + j[3])))
        gt[f] = items
    return gt


def run_tracker(dets_by_frame, fps: float, iou_thr: float, max_occluded_ms: int, drop_rate: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    tr = SimpleIoUTracker("cam1", iou_thr=iou_thr, max_occluded_ms=max_occluded_ms)
    pred: dict[int, list] = {}
    id_map: dict[str, int] = {}
    for f in sorted(dets_by_frame):
        t_ms = int(f * 1000 / fps)
        dets = [Detection(box=b, class_label="person", confidence=0.9)
                for _, b in dets_by_frame[f] if rng.random() >= drop_rate]
        live, _ = tr.update(dets, t_ms)
        pred[f] = [(id_map.setdefault(t.tube_id, len(id_map) + 1), t.box) for t in live if t.state.value == "active"]
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mot17", help="sequence dir containing gt/gt.txt and det/det.txt")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the sequence")
    ap.add_argument("--sample-every", type=int, default=1, help="use every Nth frame (simulate 2-5 fps decode)")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--max-occluded-ms", type=int, default=2000)
    ap.add_argument("--det-min-conf", type=float, default=0.5)
    ap.add_argument("--drop-rate", type=float, default=0.0, help="synthetic: random missed detections")
    a = ap.parse_args()
    if a.synthetic:
        gt = synthetic_gt()
        dets = gt
        name = "synthetic_crossing"
    elif a.mot17:
        seq = Path(a.mot17)
        gt = load_mot_txt(seq / "gt" / "gt.txt", gt=True)
        dets = load_mot_txt(seq / "det" / "det.txt", min_conf=a.det_min_conf)
        name = seq.name
    else:
        ap.error("--synthetic or --mot17 required")
    if a.sample_every > 1:
        gt = {f: v for f, v in gt.items() if f % a.sample_every == 0}
        dets = {f: v for f, v in dets.items() if f % a.sample_every == 0}
    pred = run_tracker(dets, a.fps, a.iou_thr, a.max_occluded_ms, a.drop_rate)
    res = evaluate_mot(gt, pred)
    row = {"ring": 2, "tracker": "SimpleIoUTracker", "sequence": name, "fps": a.fps, "sample_every": a.sample_every,
           "effective_fps": round(a.fps / a.sample_every, 2), "iou_thr": a.iou_thr, "max_occluded_ms": a.max_occluded_ms,
           "drop_rate": a.drop_rate, **res.as_row(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "ring2.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/README.md << 'EOF_VI'
# bench/

One script per ring. Each prints exactly one row of the benchmark table (see PLAN.md) and
checkpoints it to `data/bench/<ring>.jsonl` as it goes, so a killed Colab session loses nothing.

| script | ring | GPU | dataset | row |
|---|---|---|---|---|
| slice_cpu.py | all (stubs) | none | synthetic | proves plumbing; run first, every day |
| slice_gpu.py | 0–3a on real footage | L4 | own clip | reader → gate → ROIs → RF-DETR → tubes → events → episode file + birth keyframes; stage ms, tubes by state |
| ring0_gate.py | 0 | CPU | VIRAT | gate FN rate, ms per GOP per stream, MV vs framediff |
| ring1_detect.py | 1 | L4 | own clip / VIRAT | `--mode frame`: ms p50/p95 per frame at sampled fps; `--mode roi`: gate → packed ROI batches (nano/medium) |
| ring2_tubes.py | 2 | CPU | synthetic, MOT17 | MOTA/IDF1/IDSW/fragmentation (vi/eval/mot.py), `--sample-every` simulates 2–5 fps decode |
| ring3b_sheet.py | 3b | A100-40 | crops from ring2 | attribute accuracy, bleed rate (12 vs 16 cells), ms per sheet |
| ring3a_events.py | 3a | CPU | MEVA subset | event precision/recall per type |
| agent_replay.py | block 2 | A100-40 -> G4 | WILDTRACK episodes | both scenarios replay; tool calls, latency |

Colab: GPU runtime, `pip install rfdetr==1.7.0`, then run from the repo root (see colab/README.md).
EOF_VI
cat > tests/test_eval_mot.py << 'EOF_VI'
import pytest

from vi.eval import evaluate_mot
from vi.schemas import Box


def b(x):
    return Box(x1=x, y1=0, x2=x + 40, y2=100)


def test_perfect_tracking_scores_one():
    gt = {f: [(1, b(f * 5)), (2, b(300 - f * 5))] for f in range(20)}
    r = evaluate_mot(gt, gt)
    assert r.mota == 1.0 and r.idf1 == 1.0 and r.idsw == 0 and r.fragmentation_ratio == 1.0


def test_id_switch_is_counted_and_lowers_idf1():
    gt = {f: [(1, b(f * 5))] for f in range(20)}
    pred = {f: [((7 if f < 10 else 8), b(f * 5))] for f in range(20)}
    r = evaluate_mot(gt, pred)
    assert r.idsw == 1 and r.mota == pytest.approx(1 - 1 / 20) and r.idf1 == 0.5
    assert r.fragmentation_ratio == 2.0


def test_misses_and_false_positives():
    gt = {f: [(1, b(f * 5))] for f in range(10)}
    pred = {f: ([(1, b(f * 5))] if f % 2 == 0 else []) for f in range(10)}
    pred[3] = [(9, b(900))]
    r = evaluate_mot(gt, pred)
    assert r.fn == 5 and r.fp == 1 and r.tp == 5


@pytest.mark.edge("E-DET-09")
def test_default_zones_and_tube_classes():
    from vi.detect import is_tube_class
    from vi.events import default_zones
    z = default_zones("c1", 1280, 720)
    assert [x.zone_id for x in z] == ["exit_left", "exit_right", "floor_centre"]
    assert z[0].contains((10, 300)) and z[1].contains((1270, 300)) and z[2].contains((640, 600))
    assert is_tube_class("person") and is_tube_class("suitcase")
    assert not is_tube_class("dining table") and not is_tube_class("tv")
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
  handling: Heartbeat detections on I-frames (1–2 s active tiles, 10–30 s quiet, always on asset homes) keep tubes alive; not the gate's job.
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
  handling: Heartbeat detections catch state deltas the gate misses; gate FN rate is a tracked metric.
  status: planned
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
  handling: media_zones from scene card; detections fully inside a media zone with no tube motion are dropped.
  status: planned
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
  handling: New tube; fusion links to the same entity via ReID + gallery; the agent reports entity, not tube.
  status: planned
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
  handling: Multiple exemplars per entity; embedding refresh on confident matches; oldest expire.
  status: planned
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
  handling: dt-aware Kalman prediction (ByteTrack rewrite) and a lower IoU gate; measured as fragmentation_ratio and idsw in bench/ring2_tubes.py against the SimpleIoUTracker baseline.
  status: planned
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
  handling: Say so; propose widening; never fabricate.
  status: planned
- id: E-AGT-03
  ring: agent
  title: Unknown named entity
  trigger: '"Jay" not in gallery.'
  handling: Ask who Jay is; offer naming form.
  status: planned
- id: E-AGT-04
  ring: agent
  title: Fuzzy time expressions
  trigger: '"yesterday afternoon", "after lunch".'
  handling: Resolve to explicit local window; echo it in the answer.
  status: planned
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
  handling: verify() before asserting; every claim cites record ids and timestamps.
  status: planned
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
cat > pyproject.toml << 'EOF_VI'
[project]
name = "vi-engine"
version = "0.1.0"
description = "Video intelligence engine: tube-centric perception (Block 1) + evidence-backed reasoning (Block 2)"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.6", "numpy>=1.26", "pyyaml>=6"]

[project.optional-dependencies]
dev = ["pytest>=8"]
ingest = ["av>=13", "pillow>=10", "scipy>=1.11"]
perception = ["rfdetr==1.7.0", "av>=13", "torch", "torchreid"]
serving = ["vllm>=0.12", "psycopg[binary]>=3.2", "boto3"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vi*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["edge(id): test covers the edge case with this ID from edge_cases.yaml"]
EOF_VI
cp "$0" colab/sessions/session_04_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. Ring 2 baseline: SimpleIoUTracker on synthetic crossings (CPU)"
python bench/ring2_tubes.py --synthetic | grep -E '"(sequence|effective_fps|mota|idf1|idsw|fragmentation_ratio)"'
echo "-- same sequence sampled to 2 fps (what R11 decode looks like to the tracker)"
python bench/ring2_tubes.py --synthetic --sample-every 15 --fps 30 | grep -E '"(effective_fps|mota|idf1|idsw|fragmentation_ratio)"'
echo "-- 6 fps"
python bench/ring2_tubes.py --synthetic --sample-every 5 --fps 30 | grep -E '"(effective_fps|mota|idf1|idsw|fragmentation_ratio)"'

step "5. real-footage slice (GPU)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  if [ -f "$SOURCE" ]; then
    python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --max-frames 400
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — Runtime > Change runtime type > L4, then rerun for the slice"
fi

step "6. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: real-footage GPU slice, keyframe store, MOT evaluator, ring2 baseline bench, tube-class filter (E-DET-09)"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "7. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "bench rows: $(wc -l data/bench/*.jsonl 2>/dev/null | tail -1)"
echo "next: paste this cell's output back. Session 05 = MOT17-half on Drive + real baseline; session 06 = ByteTrack rewrite (dt-aware Kalman)."
