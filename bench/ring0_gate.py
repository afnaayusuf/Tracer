"""Ring 0 bench: reader -> FrameDiffGate over a clip. Prints one benchmark row and appends it to
data/bench/ring0.jsonl. Run: python bench/ring0_gate.py --synthetic   |   --source clip.mp4 --fps 5"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from vi.events import EventCompiler
from vi.gate import FrameDiffGate
from vi.ingest import VideoReader
from vi.ingest.synthetic import write_walk_clip


def run(source: str, camera_id: str, fps: float | None, max_width: int) -> dict:
    reader = VideoReader(camera_id, source, target_fps=fps, max_width=max_width)
    gate = FrameDiffGate(camera_id)
    compiler = EventCompiler(camera_id, [])
    gate_ms = 0.0
    frames = blobs = luma_steps = global_motion = 0
    events = []
    last_size = None
    for fr in reader.frames():
        if fr.size_changed:
            events += compiler.on_size_change(fr.pts_ms, last_size, (fr.width, fr.height))
        last_size = (fr.width, fr.height)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        gate_ms += (time.perf_counter() - t0) * 1000
        events += compiler.on_gate(g)
        frames += 1
        blobs += len(g.blobs)
        luma_steps += g.luma_step
        global_motion += g.global_motion
    st = reader.stats
    row = {
        "ring": 0, "gate": "framediff", "source": Path(source).name if not reader.is_live else "rtsp",
        "codec": st.codec, "gate_mode_recommended": st.gate_mode, "native_size": st.native_size,
        "frames_decoded": st.frames_decoded, "frames_emitted": frames,
        "dropped": {"no_pts": st.no_pts, "duplicates": st.duplicates, "out_of_order": st.out_of_order,
                    "sampled_out": st.sampled_out},
        "size_changes": st.size_changes,
        "decode_ms_per_frame": round(st.decode_ms_total / max(1, frames), 3),
        "gate_ms_per_frame": round(gate_ms / max(1, frames), 3),
        "blobs_per_frame": round(blobs / max(1, frames), 3),
        "luma_steps": luma_steps, "global_motion_frames": global_motion,
        "events": sorted({e.type.value for e in events}),
        "host": platform.processor() or platform.machine(), "python": platform.python_version(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="file path or rtsp:// url")
    ap.add_argument("--synthetic", action="store_true", help="generate data/synthetic/walk.mp4 and bench it")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--fps", type=float, default=None, help="sample to this fps (R11); default native")
    ap.add_argument("--max-width", type=int, default=640)
    args = ap.parse_args()
    if args.synthetic:
        args.source = str(write_walk_clip("data/synthetic/walk.mp4", seconds=8, fps=10, light_switch_at_s=5))
    if not args.source:
        ap.error("--source or --synthetic required")
    row = run(args.source, args.camera, args.fps, args.max_width)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "ring0.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
