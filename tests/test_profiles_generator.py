import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

from vi.generator import BRANCHES, plan
from vi.profiles import available_profiles, load_profile


def test_every_profile_loads_extends_common_and_names_known_or_declared_branches():
    names = available_profiles()
    assert "common" in names and len(names) >= 6
    common = load_profile("common")
    for n in names:
        p = load_profile(n)
        assert set(common.branches) <= set(p.branches)            # tiers add, never remove
        assert p.quality.min_height_frac == 0.4 and p.sampling["active_fps"] == 4
    t1 = load_profile("tier1_hospital")
    assert "fall" in t1.events and "enter_zone" in t1.events and "wheelchair" in t1.tube_classes


def test_plan_reports_unavailable_branches_instead_of_failing():
    p = plan(["detect", "pose", "made_up"], available_modules={"rfdetr"})
    assert p["detect"]["runnable"] and p["pose"]["runnable"] is False and "mmpose" in p["pose"]["why"]
    assert p["made_up"]["branch"] is None
    assert all(b.license and b.status in ("measured", "planned", "stub") for b in BRANCHES.values())


def test_media_zone_fires_no_events():
    from vi.events import EventCompiler, Zone
    from vi.schemas import TubeSnapshot, TubeState, Box
    z = Zone(zone_id="rack", camera_id="c1", kind="media", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    ec = EventCompiler("c1", [z], enter_ticks=1, open_grace_ms=0)
    snap = TubeSnapshot(tube_id="a", class_label="person", state=TubeState.active, box=Box(x1=10, y1=10, x2=30, y2=90))
    assert ec.on_tick([], -500) == [] and ec.on_tick([snap], 0) == [] and ec.on_tick([snap], 500) == []


def test_writer_parser_tolerates_prose_missing_cells_and_bad_colors():
    from vi.writer import parse_sheet_reply
    reply = 'Here you go:\n```json\n[{"cell_id": 0, "top_color": "ORANGE", "headwear": "white hard hat", "description": "worker at a table", "confidence": 0.9},' \
            ' {"cell_id": 1, "top_color": "neon", "carried_item": "box"}]\n```'
    r = parse_sheet_reply(reply, ["t0", "t1", "t2"])
    assert r is not None and r.expected_cells == 3 and len(r.cells) == 3
    a0, a1, a2 = (c.attributes for c in r.cells)
    assert a0.top_color.value == "orange" and "hard hat" in a0.description and a0.confidence == 0.9
    assert a1.top_color is None and a1.carried_item == "box"
    assert a2.confidence == 0.0                                      # missing cell: empty, not invented
    assert parse_sheet_reply("no json here", ["t0"]) is None


def test_pack_sheet_numbers_cells():
    import numpy as np
    from vi.writer import pack_sheet
    sheet = pack_sheet([np.zeros((120, 40, 3), np.uint8)] * 5, cell=100, cols=3)
    assert sheet.size == (300, 2 * (100 + 22))      # two rows, each cell + caption strip


def test_numeric_tools_and_whole_time_check(tmp_path):
    from vi.agent import ask, count_entities, coverage, entities_present, search_events
    from vi.store import connect, load_episode_file
    out = subprocess.run([sys.executable, "bench/slice_cpu.py", str(tmp_path / "ep")], capture_output=True, text=True)
    assert out.returncode == 0
    path = next((tmp_path / "ep").glob("*.jsonl"))
    engine = connect(); load_episode_file(engine, path)
    ep = search_events(engine)[0]["episode_id"]
    cov = coverage(engine, ep)
    assert cov and 0 < cov[0]["coverage"] <= 1
    assert count_entities(engine, ep)["count"] == 1
    assert entities_present(engine, ep, 0.99)["count"] == 0          # the walker is not present for the whole clip

    class Overcounter:
        name = "over"
        def __init__(self): self.n = 0
        def complete(self, messages, schema):
            self.n += 1
            if self.n == 1:
                return json.dumps({"action": "answer", "text": "There were 4 people.", "citations": ["cam1:2000:1"], "confidence": 0.9})
            assert "count_entities says 1" in messages[-1]["content"]
            return json.dumps({"action": "answer", "text": "There was 1 person.", "citations": ["cam1:2000:1"], "confidence": 0.9})
    res = ask(engine, "How many people were there?", ep, Overcounter())
    assert any(t.get("revise") == "numeric claim disagrees with tools" for t in res["trace"]) and "1 person" in res["final"]["text"]


def test_metamorphic_bench_runs_on_synthetic_with_fake_detector(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/metamorphic.py", "--source", str(clip), "--variants", "brightness_up,hflip,fps6",
                          "--fps", "5", "--work", str(tmp_path / "mm"), "--", "--model", "fake", "--detect", "frame"],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    assert "3/3 invariants hold" in out.stdout


def test_time_grounding_and_scope_guards():
    from datetime import datetime, timezone
    from vi.agent import classify_scope, ground_time
    start = int(datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc).timestamp() * 1000); end = start + 3600_000
    assert ground_time("Who was at the door at 12:30?", end, start, end).kind == "future"
    assert ground_time("What happened at 9:30?", end, start, end).kind == "before_start"
    g = ground_time("How many people in the last 15 minutes?", end, start, end)
    assert g.kind == "ok" and g.t_start_ms == end - 15 * 60_000
    g = ground_time("What happened between 10:10 and 10:20?", end, start, end)
    assert (g.t_start_ms - start, g.t_end_ms - start) == (10 * 60_000, 20 * 60_000)
    assert ground_time("How many people were there?", end, start, end).kind == "none"
    assert classify_scope("What's the weather?")[0] == "off_topic" and classify_scope("Lock the door")[0] == "act"
    assert classify_scope("Who was at the conveyor?")[0] == "ok" and classify_scope("Who is he?")[0] == "identity"


def test_ask_window_refuses_future_and_off_topic_without_a_model_call(tmp_path):
    from vi.agent import ask_window
    from vi.store import connect
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=8, fps=10)
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(clip), "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--out", str(tmp_path / "ep")],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1200:]
    engine = connect(db)

    class Never:
        name = "never"
        def complete(self, messages, schema):
            raise AssertionError("the model must not be called for guarded questions")
    now = int(datetime.fromisoformat("2026-09-27T10:00:10+00:00").timestamp() * 1000)
    assert ask_window(engine, "Who was at the door at 15:00?", Never(), now)["grounding"] == "future"
    assert ask_window(engine, "What's the weather like?", Never(), now)["grounding"] == "off_topic"
    assert ask_window(engine, "Unlock the door", Never(), now)["grounding"] == "act"
    from vi.agent import FakeBackend
    r = ask_window(engine, "How many people in the last 5 seconds?", FakeBackend(), now)
    assert r["grounding"] == "ok" and r["final"]["action"] == "answer" and r["window"]


def test_ui_builds_without_launching(tmp_path):
    pytest.importorskip("gradio")
    import sys as _sys
    _sys.argv = ["x"]
    from ui.app import build
    demo = build("sqlite+pysqlite:///:memory:", "fake", "fake", "UTC")
    assert demo is not None


def test_border_only_flags_brief_tubes():
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    long = Tube(tube_id="c1:0:1", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0), last_seen=CamTime(cam_utc_ms=15000),
                box=Box(x1=290, y1=80, x2=318, y2=200), max_height_px=120)
    grade_tube(long, 320, 240)
    assert long.quality == "ok"                                     # walked out through the edge after 15 s
    brief = Tube(tube_id="c1:0:2", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0), last_seen=CamTime(cam_utc_ms=2000),
                 box=Box(x1=0, y1=80, x2=20, y2=200), max_height_px=120)
    grade_tube(brief, 320, 240)
    assert brief.quality == "low" and "border" in brief.quality_reason


def test_grid_ingest_makes_a_camera_per_cell_and_camera_questions_ground(tmp_path):
    import av
    import numpy as np
    from vi.agent import ask_window, FakeBackend, footage_bounds
    from vi.store import connect
    rng = np.random.default_rng(1)
    path = tmp_path / "grid.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 30):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:
            f[cy + 4: cy + 20, cx + 250: cx + 316] = 240               # burned-in labels
        if 3 <= t < 18:
            x = 10 + int(((t - 3) / 15) * 260); f[80:200, x:x + 24] = 235            # cell 0 walker
        if 10 <= t < 28:
            x = 330 + int(((t - 10) / 18) * 260); f[320:440, x:x + 24] = 235        # cell 3 walker
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "2x2", "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "fake", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
                          "--out", str(tmp_path / "ep"), "--live-dir", str(tmp_path / "live")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    row = json.loads(out.stdout[out.stdout.rindex("{\n"):])
    assert row["cameras"] == 4 and row["episodes"] == 2 and row["realtime_factor"] > 1
    engine = connect(db)
    from vi.agent.tools import cameras_in_store, count_entities_window
    assert cameras_in_store(engine) == ["cam01", "cam04"]              # the label boxes never became tubes
    b = footage_bounds(engine)
    assert count_entities_window(engine, b[0], b[1], camera_id="cam04")["count"] == 1
    r = ask_window(engine, "Who was on cam 4?", FakeBackend(), b[1])
    assert r["camera_id"] == "cam04"
    assert ask_window(engine, "What happened on camera 9?", FakeBackend(), b[1])["grounding"] == "unknown_camera"
    assert (tmp_path / "live" / "latest.jpg").exists()


def test_grid_autodetect_and_cross_camera_fusion(tmp_path):
    """A 2x2 NVR view of ONE walker seen by all four cells at once: auto grid, and the count is 1 person / 4 tracks."""
    import av
    import numpy as np
    from vi.agent import ask_window, FakeBackend, footage_bounds
    from vi.agent.tools import count_entities_window, window_script
    from vi.store import connect
    rng = np.random.default_rng(3)
    path = tmp_path / "shop.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 24):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        f[:, 318:322] = 0; f[238:242, :] = 0                          # NVR seams
        if 3 <= t < 21:
            x = 10 + int(((t - 3) / 18) * 250)
            for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:  # the same bright figure in every cell = same person, 4 angles
                f[cy + 60: cy + 180, cx + x: cx + x + 24] = 235
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "auto", "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "none", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
                          "--out", str(tmp_path / "ep"), "--live-dir", str(tmp_path / "live")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "grid auto -> 2x2" in out.stdout
    engine = connect(db)
    b = footage_bounds(engine)
    r = count_entities_window(engine, b[0], b[1])
    assert r["camera_entities"] == 4 and r["people"] == 1 and len(r["world_groups"]) == 1        # one man, four cameras
    script = window_script(engine, b[0], b[1])
    assert "CAST: 1 people (4 camera tracks" in script and "W1: " in script
    res = ask_window(engine, "How many people were there?", FakeBackend(), b[1])
    assert res["final"]["action"] == "answer"


def test_tile_discovery_then_tile_identity_end_to_end(tmp_path):
    """Phase 1 (per camera) records one man on four cameras; discovery groups the cameras; phase 2
    with the tile map gives him ONE id at ingest time."""
    import av
    import numpy as np
    from vi.agent import footage_bounds
    from vi.agent.tools import cameras_in_store, coverage_window
    from vi.fusion import discover_tiles
    from vi.fusion.tiles import entities_for_affinity
    from vi.store import connect
    rng = np.random.default_rng(5)
    path = tmp_path / "shop.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 24):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        f[:, 318:322] = 0; f[238:242, :] = 0
        # two different-looking people, one after the other, each seen by all four cells at once
        for (t0, t1, val) in [(3, 10, 250), (13, 21, 205)]:      # both above the fake detector luma (200), different histogram bins
            if t0 <= t < t1:
                x = 10 + int(((t - t0) / (t1 - t0)) * 250)
                for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:
                    f[cy + 60: cy + 180, cx + x: cx + x + 24] = val
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    base = [sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "2x2", "--db", db, "--model", "fake", "--reid", "hist",
            "--writer", "none", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
            "--live-dir", str(tmp_path / "live")]
    out = subprocess.run(base + ["--out", str(tmp_path / "ep1"), "--tiles", "none"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    engine = connect(db)
    tm = discover_tiles(entities_for_affinity(engine), cameras=cameras_in_store(engine))
    assert list(tm.tiles.values()) == [["cam01", "cam02", "cam03", "cam04"]]            # all four see the same area
    tm.save(tmp_path / "tiles.json")
    db2 = f"sqlite+pysqlite:///{tmp_path / 'vi2.db'}"
    out = subprocess.run([*(base[:base.index('--db') + 1]), db2, *base[base.index('--db') + 2:], "--out", str(tmp_path / "ep2"),
                          "--tiles", str(tmp_path / "tiles.json")], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "BuF homo: tiles {'T1': ['cam01', 'cam02', 'cam03', 'cam04']}" in out.stdout
    e2 = connect(db2); b = footage_bounds(e2)
    rows = coverage_window(e2, b[0], b[1])
    assert len(rows) == 2 and {r["entity_id"] for r in rows} == {"site:E1", "site:E2"}  # two people, each ONE id across four cameras
    assert all(sorted(r["cameras"]) == ["cam01", "cam02", "cam03", "cam04"] for r in rows)


def test_homo_buf_gives_one_identity_from_the_start_and_adjacency_is_learned(tmp_path):
    import av
    import numpy as np
    from vi.agent import footage_bounds
    from vi.agent.tools import coverage_window
    from vi.fusion import TileMap, discover_tiles
    from vi.store import connect
    rng = np.random.default_rng(7)
    path = tmp_path / "shop.mp4"; c = av.open(str(path), "w"); s = c.add_stream("mpeg4", rate=5); s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
    for i in range(5 * 24):
        t = i / 5
        f = rng.normal(100, 3, (480, 640)).clip(0, 255).astype(np.uint8)
        f[:, 318:322] = 0; f[238:242, :] = 0
        if 3 <= t < 21:
            x = 10 + int(((t - 3) / 18) * 250)
            for (cx, cy) in [(0, 0), (320, 0), (0, 240), (320, 240)]:
                f[cy + 60: cy + 180, cx + x: cx + x + 24] = 235
        for pk in s.encode(av.VideoFrame.from_ndarray(np.repeat(f[:, :, None], 3, axis=2), format="rgb24")): c.mux(pk)
    for pk in s.encode(): c.mux(pk)
    c.close()
    db = f"sqlite+pysqlite:///{tmp_path / 'vi.db'}"
    out = subprocess.run([sys.executable, "bench/run_ingest.py", "--source", str(path), "--grid", "auto", "--db", db, "--model", "fake", "--reid", "hist",
                          "--writer", "none", "--tiles", "one", "--start-time", "2026-09-27T10:00:00+00:00", "--fps", "5", "--threshold", "0.3",
                          "--out", str(tmp_path / "ep"), "--live-dir", str(tmp_path / "live")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "BuF homo: tiles {'T1': ['cam01', 'cam02', 'cam03', 'cam04']}" in out.stdout
    engine = connect(db); b = footage_bounds(engine)
    rows = coverage_window(engine, b[0], b[1])
    assert len(rows) == 1 and rows[0]["entity_id"] == "site:E1" and sorted(rows[0]["cameras"]) == ["cam01", "cam02", "cam03", "cam04"]
    # adjacency from sequential sightings (synthetic entities): T1 -> T2 twice, ~10 s apart
    def u(seed):
        v = np.random.default_rng(seed).normal(size=48).astype(np.float32); return (v / np.linalg.norm(v)).tolist()
    ents = []
    for k, seed in enumerate((1, 2)):
        ents += [{"entity_id": f"cam01:E{k}", "camera_id": "cam01", "first_seen_ms": k * 60000, "last_seen_ms": k * 60000 + 8000, "embedding": u(seed)},
                 {"entity_id": f"cam05:E{k}", "camera_id": "cam05", "first_seen_ms": k * 60000 + 18000, "last_seen_ms": k * 60000 + 26000, "embedding": u(seed)}]
    tm = discover_tiles(ents, cameras=["cam01", "cam05"])
    assert tm.tiles == {"T1": ["cam01"], "T2": ["cam05"]} and tm.kind == "hetero"
    assert "T1|T2" in tm.adjacency and tm.adjacency["T1|T2"]["handoffs"] == 2 and tm.relation("cam01", "cam05")[0] == "adjacent"
