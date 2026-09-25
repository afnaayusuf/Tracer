"""Ring 1 bench: reader (sampled fps) -> RF-DETR on whole frames, or gate -> ROI batches -> RF-DETR.
One row per (model, mode) appended to data/bench/ring1.jsonl. GPU runtime required.

  python bench/ring1_detect.py --source /content/HI_DEF_VIDEO.mp4 --models nano,medium --mode frame
  python bench/ring1_detect.py --source /content/HI_DEF_VIDEO.mp4 --models nano --mode roi --batch 8
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi.detect import blobs_to_rois, crop_roi, pad_batch, remap_detections
from vi.detect.rfdetr import RFDETRDetector
from vi.gate import FrameDiffGate
from vi.ingest import VideoReader


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def run(source: str, size: str, mode: str, fps: float, max_frames: int, threshold: float,
        batch: int, warmup: int) -> dict:
    det = RFDETRDetector(size=size, threshold=threshold, batch_size=batch if mode == "roi" else 1)
    reader = VideoReader("cam1", source, target_fps=fps, max_width=640, want_rgb=True)
    gate = FrameDiffGate("cam1")
    lat, gate_ms, n_dets, n_rois, roi_px = [], 0.0, [], [], []
    classes: Counter = Counter()
    truncated = 0
    frames = 0
    for fr in reader.frames():
        if frames >= max_frames + warmup:
            break
        h, w = fr.rgb.shape[:2]
        if mode == "frame":
            t0 = time.perf_counter()
            dets = det.detect(fr.rgb)
            dt = (time.perf_counter() - t0) * 1000
        else:
            t0 = time.perf_counter()
            g = gate.update(fr.gray, fr.pts_ms)
            gate_ms += (time.perf_counter() - t0) * 1000
            rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride)
            dets = []
            t0 = time.perf_counter()
            for i in range(0, len(rois), batch):
                chunk = rois[i:i + batch]
                crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], batch)
                for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                    dets += remap_detections(d, r, w, h)
            dt = (time.perf_counter() - t0) * 1000
            n_rois.append(len(rois))
            roi_px += [(r.x2 - r.x1) * (r.y2 - r.y1) for r in rois]
        frames += 1
        if frames <= warmup:
            continue
        lat.append(dt)
        n_dets.append(len(dets))
        truncated += sum(d.truncated for d in dets)
        classes.update(d.class_label for d in dets)
    if not lat:
        raise SystemExit("no frames benchmarked; check --source")
    st = reader.stats
    row = {
        "ring": 1, "model": f"rf-detr-{size}", "resolution": det.resolution, "mode": mode,
        "optimized": det.optimized, "fp16": True, "batch": batch if mode == "roi" else 1,
        "source": Path(source).name, "codec": st.codec, "native_size": st.native_size,
        "sampled_fps": fps, "frames": len(lat), "threshold": threshold,
        "ms_p50": round(float(np.median(lat)), 2), "ms_p95": round(float(np.percentile(lat, 95)), 2),
        "ms_mean": round(float(np.mean(lat)), 2),
        "dets_per_frame": round(float(np.mean(n_dets)), 2), "truncated_dets": truncated,
        "classes": dict(classes.most_common(8)),
        "gpu": gpu_name(), "python": platform.python_version(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if mode == "roi":
        row.update({"gate_ms_per_frame": round(gate_ms / max(1, frames), 2),
                    "rois_per_frame": round(float(np.mean(n_rois)), 2),
                    "roi_px_mean": int(np.mean(roi_px)) if roi_px else 0,
                    "frames_with_no_roi": int(sum(1 for n in n_rois if n == 0))})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--models", default="nano,medium")
    ap.add_argument("--mode", choices=["frame", "roi"], default="frame")
    ap.add_argument("--fps", type=float, default=2.0, help="R11: sampled decode rate")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8, help="ROI mode: fixed traced batch size")
    args = ap.parse_args()
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    for size in [s.strip() for s in args.models.split(",") if s.strip()]:
        row = run(args.source, size, args.mode, args.fps, args.frames, args.threshold, args.batch, args.warmup)
        with (out / "ring1.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
