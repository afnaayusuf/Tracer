"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source /content/HI_DEF_VIDEO.mp4 --fps 4 --model nano            # ByteTracker
  python bench/slice_gpu.py --source /content/HI_DEF_VIDEO.mp4 --fps 4 --tracker simple --threshold 0.5
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

from vi.detect import blobs_to_rois, crop_roi, is_tube_class, pad_batch, remap_detections
from vi.detect.rfdetr import RFDETRDetector
from vi.episode import EpisodeWriter, KeyframeStore
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.tubes import TRACKERS


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
    detect_ms: list[float] = []
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
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride)
        dets = []
        for i in range(0, len(rois), a.batch):
            chunk = rois[i:i + a.batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], a.batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = [d for d in dets if is_tube_class(d.class_label)]
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
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
        "tracker": a.tracker, "det_threshold": a.threshold,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
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
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}")


if __name__ == "__main__":
    main()
