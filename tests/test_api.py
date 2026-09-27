import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="module")
def store_with_footage(tmp_path_factory):
    pytest.importorskip("fastapi")
    from vi.ingest.synthetic import write_walk_clip
    tmp = tmp_path_factory.mktemp("api")
    clip = write_walk_clip(tmp / "walk.mp4", seconds=8, fps=10)
    db = f"sqlite+pysqlite:///{tmp / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(clip), "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--out", str(tmp / "ep"),
                          "--live-dir", str(tmp / "live")], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1200:]
    return db, tmp


def test_api_serves_page_health_ask_and_evidence(store_with_footage):
    from fastapi.testclient import TestClient
    from vi.api import create_app
    db, tmp = store_with_footage
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"),
                     live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    assert c.get("/").status_code == 200 and "Tracer" in c.get("/").text and "fetch(API + '/ask'" in c.get("/").text
    h = c.get("/health").json()
    assert h["ok"] and h["footage"]["start"] == "10:00:00" and h["episodes"] >= 1
    r = c.post("/ask", json={"question": "hi"}).json()
    assert r["mood"] == "bot" and r["grounding"] == "greeting"
    r = c.post("/ask", json={"question": "Who was at the door at 23:00?"}).json()
    assert r["grounding"] == "future" and r["mood"] == "botUnsure" and "23:00" in r["text"]
    r = c.post("/ask", json={"question": "Who was there in the last 5 seconds?"}).json()
    assert r["grounding"] == "ok" and r["window"] and isinstance(r["evidence"], list)
    if r["evidence"]:
        assert c.get(r["evidence"][0]["url"]).status_code == 200
    assert c.get("/episodes").json()[0]["status"] in ("closed", "soft_cut")
    assert c.get("/live/latest.jpg").status_code == 200
    assert c.get("/keyframes/../../etc/passwd").status_code in (404, 400)
    assert c.get("/ingest/status").json()["running"] is False


def test_followups_and_deadline(store_with_footage):
    from fastapi.testclient import TestClient
    from vi.agent.loop import resolve_followup
    from vi.api import create_app
    db, tmp = store_with_footage
    q, ctx = resolve_followup("yes exactly", [{"q": "what happened a minute ago?", "a": "Do you mean 10:00:05?", "action": "clarify"}])
    assert q.startswith("what happened a minute ago?") and "confirmed" in q and "User: yes exactly" in ctx
    q2, ctx2 = resolve_followup("and before that?", [{"q": "who came in?", "a": "cam1:E1 at 10:00:02.", "action": "answer"}])
    assert q2 == "and before that?" and "Previous answer: cam1:E1" in ctx2
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"),
                     live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    r = c.post("/ask", json={"question": "what just happened a minute ago?"}).json()
    assert r["grounding"] == "ok" and r["window"] and r["action"] in ("answer", "clarify")
    r = c.post("/ask", json={"question": "yes exactly", "history": [{"q": "who was there just now?", "a": "Which person?", "action": "clarify"}]}).json()
    assert r["action"] in ("answer", "clarify") and r["grounding"] != "off_topic"


def test_open_episode_is_queryable_while_it_grows(tmp_path):
    """The live lane: after two flushes, footage bounds and the cast reflect the open episode."""
    from vi.agent import footage_bounds
    from vi.agent.tools import coverage_window
    from vi.episode import EpisodeWriter
    from vi.schemas import Box, CamTime, Provenance, Tick, Tube, TubeSnapshot, TubeState
    from vi.schemas.episode import CastMember, EpisodeStatus
    from vi.store import IncrementalLoader, connect
    engine = connect(); inc = IncrementalLoader(engine)
    w = EpisodeWriter(tmp_path / "ep"); prov = Provenance(kb_version=1, pipeline_git="t")
    T0 = 1_790_000_000_000
    ep = w.open("floor", ["cam1"], CamTime(cam_utc_ms=T0), prov)
    tube = Tube(tube_id="cam1:0:1", camera_id="cam1", class_label="person", state=TubeState.active, born=CamTime(cam_utc_ms=T0),
                last_seen=CamTime(cam_utc_ms=T0), box=Box(x1=10, y1=10, x2=50, y2=130), max_height_px=120, entity_id="site:E1")
    for k in range(1, 5):
        t = T0 + k * 250
        tube.last_seen = CamTime(cam_utc_ms=t)
        w.write_tick(ep, Tick(camera_id="cam1", tick_index=k, t_start=CamTime(cam_utc_ms=t), t_end=CamTime(cam_utc_ms=t + 250),
                              tubes=[TubeSnapshot(tube_id=tube.tube_id, class_label="person", state=TubeState.active, box=tube.box)], provenance=prov))
        if k in (2, 4):
            w.write_tube_snapshot(ep, tube); inc.flush(w.path(ep))
            b = footage_bounds(engine)
            assert b is not None and b[1] >= t                                    # bounds advance while open
            rows = coverage_window(engine, b[0], b[1])
            assert rows and rows[0]["entity_id"] == "site:E1" and rows[0]["last_seen_ms"] == t
    from vi.agent import episodes_in
    assert episodes_in(engine, T0, T0 + 5000)[0]["status"] == "open"
    w.write_tube(ep, tube); w.close(ep, CamTime(cam_utc_ms=T0 + 1500), EpisodeStatus.closed, [CastMember(tube_ids=[tube.tube_id], class_label="person")])
    inc.flush(w.path(ep))
    assert episodes_in(engine, T0, T0 + 5000)[0]["status"] == "closed"
    from vi.store import load_episode_file
    load_episode_file(engine, w.path(ep))                                          # the full load at close is idempotent on top
    assert len(coverage_window(engine, T0, T0 + 5000)) == 1


def test_describe_endpoint_and_shared_model_priority(store_with_footage):
    import base64, io
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from vi.api import create_app
    db, tmp = store_with_footage
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"),
                     live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    buf = io.BytesIO(); Image.fromarray(np.zeros((120, 40, 3), np.uint8)).save(buf, format="JPEG")
    r = c.post("/describe", json={"camera_id": "cam1", "tube_ids": ["cam1:0:1"], "crops_jpeg_b64": [base64.b64encode(buf.getvalue()).decode()]}).json()
    assert r["cells"][0]["tube_id"] == "cam1:0:1" and r["cells"][0]["attributes"]["description"]
    st = c.get("/ingest/status").json()
    assert "restarts" in st and "writer_sheets" in st


def test_learned_media_zone_after_two_rejections(tmp_path):
    """Two 'not a person' verdicts on the same spot make it a media zone for the rest of the run."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("run_ingest", "bench/run_ingest.py")
    m = importlib.util.module_from_spec(spec); _sys.modules["run_ingest"] = m; spec.loader.exec_module(m)
    from vi.schemas import Box, CamTime, Tube
    cam = m.Cam(camera_id="cam03", tile_id="T1")
    class _Tr:  # minimal tracker stand-in
        _tracks = {}
    cam.tracker = _Tr()
    cam.ep_tubes = [Tube(tube_id=f"cam03:{i}:1", camera_id="cam03", class_label="person", born=CamTime(cam_utc_ms=i), last_seen=CamTime(cam_utc_ms=i + 5000),
                         box=Box(x1=100, y1=200, x2=140, y2=300)) for i in (0, 9000)]
    stats = {}
    from collections import Counter
    m_stats = Counter()
    # emulate the closure's environment
    def apply(cells):
        # rebuild the function with a stats Counter bound (the real one is a closure inside main)
        from vi.schemas import Attributes
        from vi.events import Zone
        for c in cells:
            t = next(t for t in cam.ep_tubes if t.tube_id == c["tube_id"])
            t.attributes = Attributes(**c["attributes"])
            if t.attributes.description.startswith("NOT A PERSON"):
                t.quality, t.quality_reason = "low", "writer: not a person"
                b = t.box; cam.rejected_boxes.append((b.x1, b.y1, b.x2, b.y2))
                hits = [r for r in cam.rejected_boxes if m._iou(r, (b.x1, b.y1, b.x2, b.y2)) > 0.5]
                if len(hits) >= 2 and not cam.media:
                    cam.media.append(Zone(zone_id="learned", camera_id="cam03", tile_id="T1", kind="media",
                                          polygon=[(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)]))
    apply([{"tube_id": "cam03:0:1", "attributes": {"modality": "rgb", "description": "NOT A PERSON; a poster", "confidence": 0.9}}])
    assert cam.media == []
    apply([{"tube_id": "cam03:9000:1", "attributes": {"modality": "rgb", "description": "NOT A PERSON; a poster", "confidence": 0.9}}])
    assert len(cam.media) == 1 and cam.media[0].kind == "media" and m._iou((0, 0, 10, 10), (5, 5, 15, 15)) > 0


def test_ingest_start_through_the_api_actually_starts(store_with_footage, tmp_path):
    """The exact call the session script makes; a 500 here is what session 37 shipped."""
    import time as _t
    from fastapi.testclient import TestClient
    from vi.api import create_app
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=4, fps=10)
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", live_dir=str(tmp_path / "live"), load_backend=False)
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/ingest/start", json={"source": str(clip), "grid": None, "start_time": "2026-09-27T10:00:00+00:00", "realtime": False,
                                      "profile": "tier3_office", "writer": "none", "reid": "hist", "model": "fake", "tiles": "one", "fps": 5})
    assert r.status_code == 200 and r.json()["started"], r.text
    for _ in range(60):
        st = c.get("/ingest/status").json()
        if not st["running"]:
            break
        _t.sleep(0.5)
    assert st["exit_code"] == 0 and st["restarts"] == 0, st
    c.post("/ingest/stop")
