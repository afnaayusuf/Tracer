"""End-to-end CPU slice on a synthetic scene: gate -> (blob-as-detection) -> tubes -> events -> episode file.
Proves the plumbing with zero models. Run: python bench/slice_cpu.py [out_dir]"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vi.detect import Detection
from vi.episode import EpisodeWriter
from vi.events import EventCompiler, Zone
from vi.gate import FrameDiffGate
from vi.schemas import Box, CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.tubes import SimpleIoUTracker

W, H, TICK_MS = 320, 240, 500
SHELF = Zone(zone_id="shelf", camera_id="cam1", tile_id="entry", kind="asset_home", asset_id="bike_keys",
             polygon=[(200, 100), (270, 100), (270, 235), (200, 235)])
WARMUP = 4   # empty frames so the background does not contain the person
DOOR = Zone(zone_id="door", camera_id="cam1", tile_id="entry", kind="exit", polygon=[(0, 0), (30, 0), (30, 240), (0, 240)])


def synth_frame(rng, t_idx: int, keys_present: bool) -> np.ndarray:
    f = rng.normal(110, 3, (H, W)).clip(0, 255).astype(np.uint8)
    f[150:156, 220:226] = 40 if keys_present else 110           # the "keys" on the shelf
    x = 10 + (t_idx - WARMUP) * 12                                # person walks left -> right
    if WARMUP <= t_idx and x < W - 30:
        f[100:200, x:x + 24] = 230
    return f


def blobs_to_detections(gate_result) -> list[Detection]:
    return [Detection(box=b.box, class_label="person", confidence=0.6)
            for b in gate_result.blobs if b.box.height > 40]      # tiny blobs (the keys) are not persons


def main(out_dir: str = "data/episodes") -> Path:
    rng = np.random.default_rng(0)
    prov = Provenance(kb_version=1, pipeline_git="slice")
    gate = FrameDiffGate("cam1")
    tracker = SimpleIoUTracker("cam1", exit_boxes=[Box(x1=0, y1=0, x2=30, y2=240)])
    compiler = EventCompiler("cam1", [SHELF, DOOR], enter_ticks=1, exit_ticks=1, dwell_ms=1500, tile_id="entry")
    writer = EpisodeWriter(out_dir)
    ep = writer.open("entry", ["cam1"], CamTime(cam_utc_ms=0), prov)

    keys_present = True
    all_closed = []
    for i in range(44):
        t_ms = i * TICK_MS
        # the person is inside the shelf zone around ticks 20-24; the keys vanish while they are there
        if i == 22:
            keys_present = False
        frame = synth_frame(rng, i, keys_present)
        g = gate.update(frame, t_ms)
        events = compiler.on_gate(g)
        live, closed = tracker.update(blobs_to_detections(g), t_ms)
        all_closed += closed
        snaps = [TubeSnapshot(tube_id=tb.tube_id, class_label=tb.class_label, state=tb.state, box=tb.box) for tb in live]
        events += compiler.on_tick(snaps, t_ms)
        if i % 4 == 0:   # heartbeat on the asset home every 2 s
            inside = [s.tube_id for s in snaps if SHELF.contains(s.box.foot_point())]
            events += compiler.on_heartbeat("shelf", keys_present, t_ms, persons_inside=inside)
        tick = Tick(camera_id="cam1", tile_id="entry", tick_index=i, t_start=CamTime(cam_utc_ms=t_ms),
                    t_end=CamTime(cam_utc_ms=t_ms + TICK_MS), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            print(f"t={t_ms:6d}  {e.type.value:24s} zone={e.zone_id} subjects={e.subject_tube_ids} objects={e.object_ids}")
    tubes = [tr.tube for tr in tracker._tracks.values()] + all_closed
    cast = [CastMember(tube_ids=[tb.tube_id], class_label=tb.class_label) for tb in tubes]
    for tb in tubes:
        writer.write_tube(ep, tb)
    writer.close(ep, CamTime(cam_utc_ms=44 * TICK_MS), EpisodeStatus.closed, cast)
    print(f"tubes: {[(tb.tube_id, tb.state.value) for tb in tubes]}")
    path = writer.path(ep)
    n = sum(1 for _ in EpisodeWriter.read(path))
    print(f"\nepisode {ep}: {n} records -> {path}")
    return path


if __name__ == "__main__":
    main(*sys.argv[1:])
