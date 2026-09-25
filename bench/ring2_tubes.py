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
