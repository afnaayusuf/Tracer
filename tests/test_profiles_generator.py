import json
import os
import subprocess
import sys

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
    assert sheet.size == (300, 200)


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
