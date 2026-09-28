"""Long-running ingest, one or many cameras from one source:
  * a single camera (a file, paced to wall clock with --realtime, or an RTSP URL), or
  * a multiplexed NVR export (--grid RxC): every cell is a virtual camera; all cells of a frame are
    detected in ONE batched call (R12, cross-camera batching), then tracked, linked, described and
    compiled per camera; episodes are per camera and load into the store as they close.

  python bench/run_ingest.py --source /content/mall_hour.mp4 --grid 4x4 --profile tier2_public --model medium \
      --start-time "2026-09-27T10:00:00+05:30" --db "$DB_URL" --reid siglip --writer none --realtime
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vi.detect import dedupe_detections, full_frame_roi, remap_detections
from vi.episode import EpisodeWriter, KeyframeStore, annotate, should_soft_cut
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate
from vi.ingest import GridSpec, cell_ids, compose, label_zones, split
from vi.ingest import VideoReader
from vi.profiles import load_profile
from vi.reid import HistogramEmbedder, crop_for_embedding, make_embedder
from vi.schemas import Box, CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.store import IncrementalLoader, connect, load_episode_file
from vi.tubes import TRACKERS, TubeLinker, grade_tube


def parse_start(s: str | None) -> int:
    if not s:
        return int(time.time() * 1000)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Cam:
    camera_id: str
    tile_id: str
    w: int = 0
    h: int = 0
    gate: object = None
    tracker: object = None
    compiler: object = None
    linker: object = None
    zones: list = field(default_factory=list)
    media: list = field(default_factory=list)
    ep: str | None = None
    ep_t0: int = 0
    ep_cast_prev: set = field(default_factory=set)
    ep_tubes: list = field(default_factory=list)
    last_live_ms: int | None = None
    described: set = field(default_factory=set)
    pending: dict = field(default_factory=dict)
    pending_aux: dict = field(default_factory=dict)
    current: dict = field(default_factory=lambda: {"frame": None})
    last_annotated: np.ndarray | None = None
    sheet_times: list = field(default_factory=list)
    rejected_boxes: list = field(default_factory=list)
    last_activity: dict = field(default_factory=dict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="file path or rtsp:// url")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--camera", default="cam1"); ap.add_argument("--tile", default="floor")
    ap.add_argument("--grid", default=None, help="RxC, or 'auto' to detect the NVR layout from the seams in the first frames")
    ap.add_argument("--grid-margin", type=int, default=0)
    ap.add_argument("--start-time", default=None); ap.add_argument("--profile", default="common")
    ap.add_argument("--model", default="nano"); ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--reid", default="siglip"); ap.add_argument("--writer", choices=["none", "qwen", "fake", "auto", "remote"], default="auto",
                    help="remote = POST sheets to the API's shared model (never blocks the ingest); qwen = load a VLM here; auto = remote if --writer-url answers, else qwen for <= 8 cameras")
    ap.add_argument("--writer-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--writer-url", default="http://127.0.0.1:8000/describe")
    ap.add_argument("--writer-min-life-s", type=float, default=2.0, help="describe a person only after this long (flickers are not worth a sheet)")
    ap.add_argument("--writer-max-per-min", type=int, default=6, help="sheets per camera per minute")
    ap.add_argument("--activity-every-s", type=float, default=12.0, help="re-describe each confirmed person's activity this often (0 = off)")
    ap.add_argument("--zones", default=None, help="zones JSON (single camera); grid cameras get border exits + label media zones")
    ap.add_argument("--episode-min", type=float, default=10.0); ap.add_argument("--quiet-close-s", type=float, default=30.0)
    ap.add_argument("--realtime", action="store_true"); ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--max-width", type=int, default=0, help="decode width cap (0 = native); grids need native resolution")
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--live-dir", default="data/live"); ap.add_argument("--live-every", type=int, default=4)
    ap.add_argument("--flush-s", type=float, default=10.0, help="push the OPEN episodes' current state to the store this often (near-live lib)")
    ap.add_argument("--tiles", default="auto", help="one = homo BuF (every camera is the same space, one identity from the first frame); "
                    "auto = data/tiles.json if present (homo/hetero learned by /tiles/recompute) else per camera; none = per camera; or a JSON path")
    a = ap.parse_args()

    profile = load_profile(a.profile)
    tube_classes = set(profile.tube_classes)
    start_ms = parse_start(a.start_time)
    engine = connect(a.db)
    prov = Provenance(kb_version=1, pipeline_git="run_ingest")
    if a.grid and a.grid.lower() == "auto":
        from vi.ingest import detect_grid
        probe = VideoReader(a.camera, a.source, target_fps=1.0, max_width=1920, want_rgb=True)
        sample = []
        for f in probe.frames():
            sample.append(f.rgb)
            if len(sample) >= 5:
                break
        spec, ev = detect_grid(sample)
        print(f"[ingest] grid auto -> {(str(spec.rows) + 'x' + str(spec.cols)) if spec else 'single camera'}  (seam evidence {ev})", flush=True)
    else:
        spec = GridSpec.parse(a.grid, a.grid_margin) if a.grid else None
    n_cams = spec.n if spec else 1
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=(a.max_width or (1920 if spec else 640)), want_rgb=True)
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=n_cams)
    else:
        from vi.detect.rfdetr import RFDETRDetector
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=n_cams)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    aux = HistogramEmbedder() if embedder and embedder.name != "hist" else None
    vlm = None
    remote = None
    if a.writer in ("auto", "remote"):
        try:
            import urllib.request
            urllib.request.urlopen(a.writer_url.rsplit("/", 1)[0] + "/health", timeout=3)
            remote = a.writer_url
            print(f"[writer] remote -> {remote} (the API's model describes; this process never blocks on it)", flush=True)
        except Exception:
            remote = None
            if a.writer == "remote":
                print("[writer] remote unavailable; no descriptions", flush=True); a.writer = "none"
    if a.writer == "auto" and remote is None:
        a.writer = "qwen" if n_cams <= 8 else "none"
        print(f"[writer] auto -> {a.writer} ({n_cams} camera(s))", flush=True)
    if remote is not None:
        import base64, io, queue, threading
        from PIL import Image
        _q: "queue.Queue" = queue.Queue(maxsize=32)
        _results: "queue.Queue" = queue.Queue()
        def _worker():
            import urllib.request
            while True:
                item = _q.get()
                if item is None:
                    return
                cam_id, tids, crops, mode = item
                try:
                    payload = {"camera_id": cam_id, "tube_ids": tids, "crops_jpeg_b64": [], "mode": mode}
                    for c in crops:
                        buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(c)).save(buf, format="JPEG", quality=85)
                        payload["crops_jpeg_b64"].append(base64.b64encode(buf.getvalue()).decode())
                    req = urllib.request.Request(remote, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    r = json.load(urllib.request.urlopen(req, timeout=120))
                    _results.put((cam_id, r.get("cells", []), mode))
                except Exception as e:
                    _results.put((cam_id, {"error": str(e)[:120]}, mode))
        threading.Thread(target=_worker, daemon=True).start()
        class _Remote:
            calls = 0; last_ms = 0.0
            def submit(self, cam_id, crops, tids, mode="appearance"):
                try:
                    _q.put_nowait((cam_id, list(tids), list(crops), mode)); self.calls += 1; return True
                except queue.Full:
                    return False
            def poll(self):
                out = []
                while True:
                    try:
                        out.append(_results.get_nowait())
                    except queue.Empty:
                        return out
        vlm = _Remote()
    if a.writer == "qwen":
        try:
            from vi.writer import WriterVLM
            vlm = WriterVLM(model_id=a.writer_model); print(f"[writer] {a.writer_model} loaded", flush=True)
        except Exception as e:
            print(f"[writer] unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
    elif a.writer == "fake":
        from vi.writer.contact_sheet import parse_sheet_reply
        class _F:
            calls = 0; last_ms = 0.0
            def describe(self, crops, ids, modality=None):
                self.calls += 1
                return parse_sheet_reply(json.dumps([{"cell_id": i, "top_color": "orange", "description": "person", "confidence": 0.7} for i in range(len(ids))]), ids)
        vlm = _F()
    writer = EpisodeWriter(a.out)
    inc = IncrementalLoader(engine)
    last_flush_wall = time.time()
    ids = cell_ids(spec) if spec else [a.camera]
    from vi.fusion import TileMap
    tilemap = None
    if a.tiles == "one":
        tilemap = TileMap.homo(ids)
        tilemap.save(Path(a.live_dir) / "tiles.json")
    elif a.tiles and a.tiles != "none":
        tilemap = TileMap.load((Path(a.live_dir) / "tiles.json") if a.tiles == "auto" else a.tiles)
    def tile_for(cid: str) -> str:
        if tilemap is not None:
            return tilemap.tile_of(cid)
        return cid if spec else a.tile
    cams = {cid: Cam(camera_id=cid, tile_id=tile_for(cid)) for cid in ids}
    site_linker = TubeLinker("person", relation=tilemap.relation) if (tilemap is not None and embedder) else None
    if tilemap is not None:
        print(f"[ingest] BuF {tilemap.kind}: tiles {tilemap.tiles}" + (f" adjacency {list(tilemap.adjacency)}" if tilemap.adjacency else ""), flush=True)
    tick_ms = int(1000 / a.fps)
    frames = 0; t_wall0 = time.time(); pts0 = None
    stats: Counter = Counter()
    detect_ms: list[float] = []

    def init_cam(cam: Cam, w: int, h: int) -> None:
        cam.w, cam.h = w, h
        if spec:
            cam.zones = [z for z in default_zones(cam.camera_id, w, h, tile_id=cam.tile_id) if z.kind == "exit"]
            cam.zones += label_zones(spec, w, h, cam.camera_id, cam.tile_id)
        else:
            zp = a.zones or (str(Path("data/zones") / (Path(a.source).stem + ".json")) if (Path("data/zones") / (Path(a.source).stem + ".json")).exists() else None)
            cam.zones = load_zones(zp, cam.camera_id) if zp else default_zones(cam.camera_id, w, h, tile_id=cam.tile_id)
        cam.media = [z for z in cam.zones if z.kind == "media"]
        exits = [Box(x1=min(p[0] for p in z.polygon), y1=min(p[1] for p in z.polygon), x2=max(p[0] for p in z.polygon), y2=max(p[1] for p in z.polygon))
                 for z in cam.zones if z.kind == "exit"]
        cam.gate = FrameDiffGate(cam.camera_id)
        cam.tracker = TRACKERS["byte"](cam.camera_id, exit_boxes=exits, keyframe_sink=kf.make_sink(lambda c=cam: c.current["frame"]))
        cam.compiler = EventCompiler(cam.camera_id, cam.zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=cam.tile_id)
        cam.linker = (site_linker if site_linker is not None else TubeLinker(cam.camera_id)) if embedder else None

    def close_episode(cam: Cam, t_end_ms: int, status: EpisodeStatus) -> None:
        if cam.ep is None:
            return
        heights = sorted(t.max_height_px for t in cam.ep_tubes if t.class_label == "person" and t.max_height_px > 0)
        med = heights[len(heights) // 2] if heights else None
        for t in cam.ep_tubes:
            if cam.linker is not None and t.class_label == "person":
                t.entity_id = cam.linker.entity_of(t.tube_id) or t.entity_id
                t.embedding = cam.linker.embedding_of(t.tube_id)
            grade_tube(t, cam.w, cam.h, median_height_px=med, min_life_ms=int(profile.quality.min_life_s * 1000),
                       min_height_frac=profile.quality.min_height_frac, border_px=profile.quality.border_px)
            writer.write_tube(cam.ep, t)
        cast = [CastMember(entity_id=t.entity_id, tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0]) for t in cam.ep_tubes]
        writer.close(cam.ep, CamTime(cam_utc_ms=t_end_ms), status, cast)
        counts = load_episode_file(engine, writer.path(cam.ep))
        stats["episodes"] += 1
        print(f"[episode] {cam.camera_id} {cam.ep} {status.value} {len(cam.ep_tubes)} tubes -> store tubes={counts['tubes']} events={counts['events']}", flush=True)
        cam.ep, cam.ep_tubes, cam.ep_cast_prev = None, [], set()

    def apply_descriptions(cam: Cam, cells: list[dict], mode: str = "appearance") -> None:
        """Attach writer output to the tubes it describes (live or already closed in this episode). A
        'not a person' verdict grades the tube low; two rejections at the same spot make a media zone.
        Activity results are also written as time-stamped patches: the person's activity timeline."""
        from vi.schemas import Attributes, EnrichmentPatch
        from vi.events import Zone
        index = {tr.tube.tube_id: tr.tube for tr in cam.tracker._tracks.values()} if cam.tracker else {}
        index.update({t.tube_id: t for t in cam.ep_tubes})
        for c in cells:
            t = index.get(c.get("tube_id"))
            if t is None:
                continue
            try:
                attrs = Attributes(**c["attributes"])
            except Exception:
                continue
            if attrs.confidence <= 0:
                continue
            if mode == "activity":
                if t.attributes is not None:                        # keep appearance, add the activity fields
                    keep = t.attributes.model_dump()
                    for k in ("activity", "objects_nearby", "attention", "posture"):
                        keep[k] = getattr(attrs, k)
                    if attrs.carried_item: keep["carried_item"] = attrs.carried_item
                    try: attrs = Attributes(**keep)
                    except Exception: pass
                if cam.ep is not None:
                    t_now = cam.last_live_ms or 0
                    patch = EnrichmentPatch(patch_id=f"pt_{t.tube_id}_{t_now}", tube_id=t.tube_id, produced_at_ms=t_now, source="vlm:activity",
                                            payload={"activity": attrs.activity, "objects_nearby": attrs.objects_nearby, "attention": attrs.attention,
                                                     "posture": attrs.posture, "carried_item": attrs.carried_item, "entity_id": t.entity_id, "camera_id": cam.camera_id},
                                            confidence=attrs.confidence)
                    writer.write_patch(cam.ep, patch); stats["activity_patches"] += 1
            t.attributes = attrs
            if (attrs.description or "").startswith("NOT A PERSON"):
                t.quality, t.quality_reason = "low", "writer: not a person"
                stats["writer_rejections"] += 1
                b = t.box
                cam.rejected_boxes.append((b.x1, b.y1, b.x2, b.y2))
                hits = [r for r in cam.rejected_boxes if _iou(r, (b.x1, b.y1, b.x2, b.y2)) > 0.5]
                if len(hits) >= 2 and not any(z.zone_id == f"learned_media_{int(b.x1)}_{int(b.y1)}" for z in cam.media):
                    pad = 0.1 * max(b.width, b.height)
                    z = Zone(zone_id=f"learned_media_{int(b.x1)}_{int(b.y1)}", camera_id=cam.camera_id, tile_id=cam.tile_id, kind="media",
                             polygon=[(b.x1 - pad, b.y1 - pad), (b.x2 + pad, b.y1 - pad), (b.x2 + pad, b.y2 + pad), (b.x1 - pad, b.y2 + pad)])
                    cam.media.append(z); cam.zones.append(z); stats["learned_media_zones"] += 1
                    print(f"[media] {cam.camera_id}: learned media zone at ({int(b.x1)},{int(b.y1)})-({int(b.x2)},{int(b.y2)}) after {len(hits)} 'not a person' verdicts", flush=True)

    def step_cam(cam: Cam, rgb: np.ndarray, gray: np.ndarray, pts_ms: int, t_ms: int, dets: list) -> None:
        cam.current["frame"] = rgb
        g = cam.gate.update(gray, pts_ms)
        events = cam.compiler.on_gate(g)
        dets = dedupe_detections([d for d in dets if d.class_label in tube_classes])
        if cam.media:
            dets = [d for d in dets if not any(z.contains(d.box.foot_point()) for z in cam.media)]
        live, closed = cam.tracker.update(dets, t_ms, det_source="heartbeat")
        if cam.ep is None and live:
            cam.ep = writer.open(cam.tile_id, [cam.camera_id], CamTime(cam_utc_ms=t_ms), prov); cam.ep_t0 = t_ms
            cam.compiler = EventCompiler(cam.camera_id, cam.zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=cam.tile_id)
        if live:
            cam.last_live_ms = t_ms
        if cam.linker is not None:
            due = [t for t in live if t.class_label == "person" and t.tube_id not in cam.pending and t.tube_id not in cam.described and cam.linker.entity_of(t.tube_id) is None]
            if due:
                crops = [crop_for_embedding(rgb, t.box) for t in due]
                embs = embedder.embed(crops); auxs = aux.embed(crops) if aux else [None] * len(crops)
                for t, e, ax in zip(due, embs, auxs):
                    cam.pending[t.tube_id] = e
                    if ax is not None: cam.pending_aux[t.tube_id] = ax
            for t in live:
                if t.class_label == "person" and t.tube_id not in cam.pending: cam.linker.on_state(t, t_ms)
            for t in live:
                if t.tube_id in cam.pending and t.state.value == "active":
                    ev = cam.linker.on_birth(t, cam.pending.pop(t.tube_id), t_ms, cam.pending_aux.pop(t.tube_id, None))
                    if ev is not None: events.append(ev); stats["relinks"] += 1
            live_ids = {t.tube_id for t in live}
            for tid in [k for k in cam.pending if k not in live_ids]:
                cam.pending.pop(tid); cam.pending_aux.pop(tid, None)
            for ghost in cam.linker.absorbed:
                dead = cam.tracker.drop(ghost)
                if dead is not None: closed.append(dead)
            cam.linker.absorbed.clear()
            live = [t for t in live if t.tube_id in cam.tracker._tracks]
            for t in closed:
                mev = cam.linker.on_close(t, t_ms)
                if mev is not None: events.append(mev); stats["merges"] += 1
            for t in live:
                if t.class_label == "person": t.entity_id = cam.linker.entity_of(t.tube_id)
        if vlm is not None and frames % 4 == 0:
            recent = [x for x in cam.sheet_times if t_ms - x < 60_000]; cam.sheet_times = recent
            todo = [t for t in live if t.class_label == "person" and t.state.value == "active" and t.tube_id not in cam.described
                    and t.last_seen.corrected_ms() - t.born.corrected_ms() >= a.writer_min_life_s * 1000][:8]
            if todo and len(recent) < a.writer_max_per_min:
                crops = [crop_for_embedding(rgb, t.box, pad=0.15) for t in todo]
                if hasattr(vlm, "submit"):                                   # remote: post and move on
                    if vlm.submit(cam.camera_id, crops, [t.tube_id for t in todo]):
                        for t in todo: cam.described.add(t.tube_id)
                        cam.sheet_times.append(t_ms); stats["sheets"] += 1
                else:
                    res = vlm.describe(crops, [t.tube_id for t in todo])
                    cells = [{"tube_id": c.tube_id, "attributes": c.attributes.model_dump(mode="json")} for c in res.cells] if res is not None else []
                    apply_descriptions(cam, cells)
                    for t in todo: cam.described.add(t.tube_id)
                    cam.sheet_times.append(t_ms); stats["sheets"] += 1
        # activity at cadence: every confirmed person, wider crop with context
        if vlm is not None and a.activity_every_s and hasattr(vlm, "submit") and frames % 4 == 2:
            due = [t for t in live if t.class_label == "person" and t.state.value == "active" and t.tube_id in cam.described
                   and t_ms - cam.last_activity.get(t.tube_id, -10**12) >= a.activity_every_s * 1000][:6]
            if due and vlm.submit(cam.camera_id, [crop_for_embedding(rgb, t.box, pad=0.6) for t in due], [t.tube_id for t in due], "activity"):
                for t in due: cam.last_activity[t.tube_id] = t_ms
                stats["activity_sheets"] += 1
        if vlm is not None and hasattr(vlm, "poll"):
            for cam_id, cells, mode in vlm.poll():
                if isinstance(cells, dict):
                    stats["writer_errors"] += 1; continue
                apply_descriptions(cams[cam_id], cells, mode)
        cam.ep_tubes += closed
        if cam.ep is not None:
            snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                                  det_source="detector" if t.state.value == "active" else "predicted") for t in live]
            events += cam.compiler.on_tick(snaps, t_ms)
            writer.write_tick(cam.ep, Tick(camera_id=cam.camera_id, tile_id=cam.tile_id, tick_index=frames, t_start=CamTime(cam_utc_ms=t_ms),
                                           t_end=CamTime(cam_utc_ms=t_ms + tick_ms), tubes=snaps, event_ids=[e.event_id for e in events], provenance=prov))
            for e in events:
                writer.write_event(cam.ep, e); stats["events"] += 1
            cast_now = {t.entity_id or t.tube_id for t in live if t.class_label == "person"}
            quiet = cam.last_live_ms is not None and not live and t_ms - cam.last_live_ms > a.quiet_close_s * 1000
            if quiet or should_soft_cut(cam.ep_cast_prev, cast_now, t_ms - cam.ep_t0, max_duration_ms=int(a.episode_min * 60_000)):
                cam.ep_tubes += list(live)
                close_episode(cam, t_ms, EpisodeStatus.closed if quiet else EpisodeStatus.soft_cut)
            elif frames % 40 == 0:
                cam.ep_cast_prev = cast_now
        if a.live_every and frames % a.live_every == 0:
            cam.last_annotated = annotate(rgb, [], dets, live, f"{cam.camera_id} {datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime('%H:%M:%S')}Z live {len(live)}", None)
        cam_live[cam.camera_id] = len(live)

    cam_live: dict[str, int] = {}

    def flush_open() -> None:
        """near-live: current tube state of every open episode into its file, new lines into the store"""
        n = 0
        for cam in cams.values():
            if cam.ep is None or cam.tracker is None:
                continue
            for tr in cam.tracker._tracks.values():
                t = tr.tube
                if t.class_label == "person" and cam.linker is not None:
                    t.entity_id = cam.linker.entity_of(t.tube_id) or t.entity_id
                    t.embedding = cam.linker.embedding_of(t.tube_id)
                if t.class_label == "person":
                    grade_tube(t, cam.w, cam.h, min_life_ms=int(profile.quality.min_life_s * 1000), min_height_frac=profile.quality.min_height_frac,
                               border_px=profile.quality.border_px)
                writer.write_tube_snapshot(cam.ep, t)
            n += 1
            try:
                inc.flush(writer.path(cam.ep))
            except Exception as e:
                print(f"[flush] {cam.camera_id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        stats["flushes"] += 1 if n else 0

    fr = None
    for fr in reader.frames():
        if pts0 is None:
            pts0 = fr.pts_ms
        if a.max_minutes and fr.pts_ms - pts0 > a.max_minutes * 60_000:
            break
        if a.realtime:
            lag = (fr.pts_ms - pts0) / 1000 - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(min(lag, 1.0))
        t_ms = start_ms + fr.pts_ms
        cells = split(fr.rgb, spec) if spec else [fr.rgb]
        grays = [c.mean(axis=2).astype(np.uint8) for c in cells] if spec else [fr.gray]
        for cid, cell in zip(ids, cells):
            if cams[cid].tracker is None:
                init_cam(cams[cid], cell.shape[1], cell.shape[0])
        t0 = time.perf_counter()
        batched = det.detect_batch(cells)                    # one call for every camera in the frame (R12)
        detect_ms.append((time.perf_counter() - t0) * 1000)
        for cid, cell, gray, dets in zip(ids, cells, grays, batched):
            h, w = cell.shape[:2]
            dets = remap_detections(list(dets), full_frame_roi(w, h), w, h)
            step_cam(cams[cid], cell, gray, fr.pts_ms, t_ms, dets)
        if a.live_every and frames % a.live_every == 0:
            live_dir = Path(a.live_dir); live_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            panels = [cams[cid].last_annotated for cid in ids if cams[cid].last_annotated is not None]
            if panels:
                img = compose(panels, spec) if spec else panels[0]
                Image.fromarray(img).save(live_dir / "latest.tmp.jpg", quality=80)
                try:
                    (live_dir / "latest.tmp.jpg").replace(live_dir / "latest.jpg")
                except Exception:
                    pass
            (live_dir / "status.json").write_text(json.dumps({"frames": frames, "footage_s": round((fr.pts_ms - pts0) / 1000, 1),
                                                              "wall_s": round(time.time() - t_wall0, 1), "cameras": n_cams,
                                                              "live_tubes": sum(cam_live.values()), "per_camera": cam_live, "now_ms": t_ms,
                                                              "episodes": stats["episodes"], "sheets": stats["sheets"],
                                                              "detect_ms_p50": round(float(np.median(detect_ms[-50:])), 1) if detect_ms else None}))
        if a.flush_s and time.time() - last_flush_wall >= a.flush_s:
            flush_open(); last_flush_wall = time.time()
        frames += 1
        if frames % 200 == 0:
            el = time.time() - t_wall0
            print(f"[ingest] {frames} frames x {n_cams} cams  footage {(fr.pts_ms - pts0)/1000:6.0f}s  wall {el:6.0f}s  "
                  f"realtime x{((fr.pts_ms - pts0)/1000) / max(el, 1e-6):.2f}  detect {np.median(detect_ms[-50:]):.0f}ms/batch  live {sum(cam_live.values())}  {dict(stats)}", flush=True)
    if fr is not None:
        for cam in cams.values():
            if cam.ep is not None:
                cam.ep_tubes += [tr.tube for tr in cam.tracker._tracks.values()]
                close_episode(cam, start_ms + fr.pts_ms + tick_ms, EpisodeStatus.closed)
    footage_s = round((fr.pts_ms - (pts0 or 0)) / 1000, 1) if fr is not None else 0
    wall = round(time.time() - t_wall0, 1)
    row = {"ring": "ingest", "source": Path(a.source).name, "cameras": n_cams, "grid": a.grid, "model": a.model, "frames": frames,
           "footage_s": footage_s, "wall_s": wall, "realtime_factor": round(footage_s / max(wall, 1e-6), 2),
           "detect_ms_p50_per_batch": round(float(np.median(detect_ms)), 1) if detect_ms else None,
           "camera_frames_per_s": round(frames * n_cams / max(wall, 1e-6), 1), **dict(stats),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path("data/bench").mkdir(parents=True, exist_ok=True)
    with open("data/bench/ingest.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
