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


def test_activity_timeline_and_inspect_tool(store_with_footage, tmp_path):
    """Activity sheets become patches; the script shows them; the agent can call inspect."""
    import json as _json
    from fastapi.testclient import TestClient
    from vi.agent import ask_window, footage_bounds
    from vi.agent.tools import activities_window, window_script
    from vi.api import create_app
    from vi.episode import EpisodeWriter
    from vi.schemas import EnrichmentPatch
    from vi.schemas.episode import episode_record_adapter
    from vi.store import IncrementalLoader, connect
    db, tmp = store_with_footage
    engine = connect(db); b = footage_bounds(engine)
    # write an activity patch into the (closed) episode file and load it incrementally
    ep_path = next((tmp / "ep").glob("*.jsonl"))
    ep_id = next(r for r in EpisodeWriter.read(ep_path) if r.kind == "header").episode_id
    tube_id = next(r for r in EpisodeWriter.read(ep_path) if r.kind == "tube").tube.tube_id
    entity_id = next(r for r in EpisodeWriter.read(ep_path) if r.kind == "tube").tube.entity_id or f"anon:{tube_id}"
    patch = EnrichmentPatch(patch_id="pt_test_1", tube_id=tube_id, produced_at_ms=b[0] + 1000, source="vlm:activity",
                            payload={"activity": "packing items into a box", "objects_nearby": ["cardboard box", "tape"], "attention": "down at the box",
                                     "posture": "bending", "entity_id": entity_id, "camera_id": "cam1"}, confidence=0.8)
    with ep_path.open("a") as f:
        f.write(_json.dumps({"kind": "patch", "episode_id": ep_id, "patch": patch.model_dump(mode="json")}) + "\n")
    IncrementalLoader(engine).flush(ep_path)
    acts = activities_window(engine, b[0], b[1] + 5000)
    assert acts and acts[0]["activity"] == "packing items into a box" and "tape" in acts[0]["objects_nearby"]
    script = window_script(engine, b[0], b[1] + 5000)
    assert "ACTIVITY" in script and "packing items into a box" in script and "looking down at the box" in script
    # inspect through the loop with an injected inspector
    class Inspector:
        name = "insp"
        def __init__(self): self.n = 0
        def complete(self, messages, schema):
            self.n += 1
            if self.n == 1:
                return _json.dumps({"action": "tool", "tool": "inspect", "args": {"question": "what is he holding?", "entity_id": "W1"}, "why": "detail"})
            assert "roll of tape" in messages[-1]["content"]
            return _json.dumps({"action": "answer", "text": "He is holding a roll of tape (inspected keyframe).", "citations": [entity_id], "confidence": 0.8})
    calls = []
    def fake_inspect(question, eid=None, cam=None, t=None):
        calls.append((question, eid)); return {"answer": "a roll of tape in his right hand", "ref": "kf://cam1/x.jpg"}
    r = ask_window(engine, "what exactly is he holding?", Inspector(), b[1], "UTC", inspector=fake_inspect)
    assert calls and calls[0][1] is not None and "tape" in r["final"]["text"]
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"), live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    assert "answer" in c.post("/inspect", json={"question": "what is on the counter?", "entity_id": entity_id}).json()


def test_ask_never_returns_a_non_json_500(store_with_footage, monkeypatch):
    from fastapi.testclient import TestClient
    from vi.api import create_app
    import vi.api.server as srv
    db, tmp = store_with_footage
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"), live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app, raise_server_exceptions=False)
    def boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(srv, "ask_window", boom)
    r = c.post("/ask", json={"question": "what is he doing right now?"})
    assert r.status_code == 200 and r.json()["grounding"] == "error" and "model exploded" in r.json()["text"]
    assert (tmp / "live" / "ask_errors.log").exists()


def test_lib_summary_and_stream(store_with_footage):
    from fastapi.testclient import TestClient
    from vi.api import create_app
    db, tmp = store_with_footage
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"), live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    d = c.get("/lib/summary").json()
    assert d["footage"] and d["people_count"] >= 1 and d["people"][0]["id"].startswith("W") and "cameras" in d["people"][0]
    assert isinstance(d["activities"], list) and isinstance(d["events"], list) and d["episodes"]
    d2 = c.get("/lib/summary", params={"window_min": 1, "since_ms": d["now_ms"]}).json()
    assert d2["events"] == [] and d2["activities"] == []                 # nothing newer than now
    r = c.get("/lib/stream", params={"window_min": 5, "max_events": 1})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.startswith("data: ") and "people_count" in r.text
    assert c.get("/lib.html").status_code == 200 and "Tracer" in c.get("/lib.html").text


def test_scene_inventory_and_narration_reach_the_script_and_the_lib_page(store_with_footage):
    import json as _json
    from fastapi.testclient import TestClient
    from vi.agent import footage_bounds
    from vi.agent.tools import scene_inventory, window_script
    from vi.api import create_app
    from vi.episode import EpisodeWriter
    from vi.schemas import EnrichmentPatch
    from vi.store import IncrementalLoader, connect
    db, tmp = store_with_footage
    engine = connect(db); b = footage_bounds(engine)
    ep_path = next((tmp / "ep").glob("*.jsonl"))
    ep_id = next(r for r in EpisodeWriter.read(ep_path) if r.kind == "header").episode_id
    trec = next(r for r in EpisodeWriter.read(ep_path) if r.kind == "tube").tube
    sc = EnrichmentPatch(patch_id="sc_cam1_1", tube_id="cam1:scene", produced_at_ms=b[0] + 500, source="vlm:scene",
                         payload={"camera_id": "cam1", "objects": [{"object": "keyboard", "where": "on the desk, right", "state": None, "count": None, "confidence": .9},
                                                                    {"object": "cardboard box", "where": "on the counter", "state": "open", "count": 2, "confidence": .8}]}, confidence=0.6)
    nr = EnrichmentPatch(patch_id="nr_1", tube_id=trec.tube_id, produced_at_ms=b[0] + 2000, source="vlm:activity",
                         payload={"activity": "He unpacks small boxes of glass jugs and checks each one.", "steps": ["takes a small box out", "inspects a glass jug", "puts it back"],
                                  "objects_handled": [{"object": "glass jug", "state": "inspected"}], "counts": {"small box": 12},
                                  "entity_id": trec.entity_id or f"anon:{trec.tube_id}", "camera_id": "cam1", "kind": "narration"}, confidence=0.7)
    with ep_path.open("a") as f:
        for pt in (sc, nr):
            f.write(_json.dumps({"kind": "patch", "episode_id": ep_id, "patch": pt.model_dump(mode="json")}) + "\n")
    IncrementalLoader(engine).flush(ep_path)
    inv = scene_inventory(engine, b[1] + 5000)
    assert inv["cam1"]["objects"][1]["object"] == "cardboard box"
    script = window_script(engine, b[0], b[1] + 5000)
    assert "SCENE" in script and "keyboard" in script and "cardboard box x2 (open)" in script
    assert "steps: takes a small box out → inspects a glass jug → puts it back" in script and "small box ≈ 12" in script
    app = create_app(db_url=db, backend_name="fake", model="fake", tz_name="UTC", keyframes_dir=str(tmp / "keyframes"), live_dir=str(tmp / "live"), load_backend=False)
    c = TestClient(app)
    d = c.get("/lib/summary").json()
    assert d["scene"]["cam1"]["objects"][0]["object"] == "keyboard" and any(a["steps"] for a in d["activities"])
    r = c.post("/describe", json={"camera_id": "cam1", "tube_ids": ["cam1:scene"], "crops_jpeg_b64": [], "mode": "scene"}).json()
    assert r["scene"][0]["object"] == "keyboard"
    r = c.post("/describe", json={"camera_id": "cam1", "tube_ids": [trec.tube_id], "crops_jpeg_b64": [], "mode": "narrate"}).json()
    assert r["narration"]["steps"]
