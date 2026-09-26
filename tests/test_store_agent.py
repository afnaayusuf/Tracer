import subprocess
import sys

import pytest

from vi.agent import clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


@pytest.fixture
def episode_path(tmp_path):
    out = subprocess.run([sys.executable, "bench/slice_cpu.py", str(tmp_path / "episodes")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return next((tmp_path / "episodes").glob("*.jsonl"))


@pytest.mark.edge("E-STO-02")
def test_loader_is_idempotent_and_derives_entities(episode_path):
    engine = connect()
    first = load_episode_file(engine, episode_path)
    second = load_episode_file(engine, episode_path)
    assert first["ticks"] > 0 and first["events"] > 0 and first["tubes"] >= 1 and first["entities"] >= 1
    assert all(second[k] == 0 for k in ("ticks", "events", "tubes", "entities"))       # nothing duplicated
    ents = search_entities(engine)
    assert ents and ents[0]["entity_id"].startswith("anon:")                             # no ReID in the CPU slice


def test_search_and_clip_return_citable_ids(episode_path):
    engine = connect()
    load_episode_file(engine, episode_path)
    pick = search_events(engine, event_type="pickup")
    assert len(pick) == 1 and pick[0]["object_ids"] == ["bike_keys"] and pick[0]["event_id"].startswith("ev_")
    suspect = pick[0]["subject_tube_ids"][0]
    tubes_ = search_tubes(engine, class_label="person")
    assert any(t["tube_id"] == suspect for t in tubes_)
    ev = clip(engine, tube_id=suspect)
    assert ev["tube_ids"] == [suspect] and ev["segments"][0]["camera_id"] == "cam1"
    window = search_events(engine, t_start_ms=9000, t_end_ms=11000)
    assert {e["type"] for e in window} <= {"enter_zone", "dwell", "exit_zone"} and window


def test_scene_script_is_deterministic_and_cites(episode_path):
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    s1 = get_script(engine, ep, cache=False)
    s2 = get_script(engine, ep, cache=True)
    s3 = get_script(engine, ep, cache=True)
    assert s1 == s2 == s3 and s1.startswith("EPISODE ") and "CAST:" in s1 and "TIMELINE:" in s1
    assert "pickup" in s1 and "object bike_keys" in s1 and "[ev_" in s1
    custody_line = next(l for l in s1.splitlines() if "pickup" in l)
    assert "anon:cam1:" in custody_line          # subject cited by entity + tube


@pytest.mark.edge("E-AGT-06")
def test_agent_loop_cites_only_ids_it_was_shown(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    out = ask(engine, "Who took the bike keys?", ep, FakeBackend())
    assert out["final"]["action"] == "answer" and out["final"]["cited"]
    assert any(c.startswith("ev_") for c in out["final"]["citations"])
    assert [t["tool"] for t in out["trace"] if "tool" in t] == ["search_events"]     # script was inlined
    bad = ask(engine, "Who took the bike keys?", ep, FakeBackend(bogus_citation=True))
    revise = next(t for t in bad["trace"] if "revise" in t)     # asked to revise once, naming the bad id
    assert revise["invalid"] == ["ev_deadbeefdeadbeef"] and bad["final"]["cited"] is False


@pytest.mark.edge("E-AGT-02")
def test_agent_loop_reports_no_results_honestly(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    out = ask(engine, "Did anybody fall? (nobody did)", ep, FakeBackend())
    assert out["final"]["action"] == "answer" and "No matching" in out["final"]["text"] and out["final"]["citations"] == []


@pytest.mark.edge("E-ING-06")
def test_every_millisecond_column_is_64_bit_on_postgres():
    """UTC ms (~1.7e12) overflow PostgreSQL INTEGER; SQLite would never notice."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable
    from vi.store.db import metadata
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for col in table.columns:
            if col.name.endswith("_ms"):
                assert f"{col.name} BIGINT" in ddl, f"{table.name}.{col.name} must be BIGINT"


def test_store_round_trips_a_utc_millisecond_timestamp():
    import time
    from vi.store import connect
    from vi.store.db import insert_ignore, scripts
    from sqlalchemy import select
    engine = connect()
    now = int(time.time() * 1000)
    with engine.begin() as conn:
        insert_ignore(conn, scripts, [dict(episode_id="ep_x", text="t", rendered_at_ms=now)])
    with engine.connect() as conn:
        assert conn.execute(select(scripts.c.rendered_at_ms)).scalar() == now


def test_extract_json_tolerates_fences_prose_and_nesting():
    from vi.agent import extract_json
    assert extract_json('```json\n{"action":"answer","text":"hi {x}","citations":[]}\n```') == '{"action":"answer","text":"hi {x}","citations":[]}'
    assert extract_json('Sure! Here it is: {"action":"tool","tool":"clip","args":{"entity_id":"cam1:E1"}} thanks') \
        == '{"action":"tool","tool":"clip","args":{"entity_id":"cam1:E1"}}'
    assert extract_json('{"a": "brace in string }"}') == '{"a": "brace in string }"}'
    assert extract_json("no json here") == "no json here"


def test_transformers_backend_pipeline_with_a_stub_model():
    """The loader/generate/decode path, with a stub standing in for the 4B model."""
    import numpy as np
    from vi.agent.loop import TransformersBackend

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            assert msgs[0]["role"] == "system" and "JSON object only" in msgs[0]["content"]
            return {"input_ids": np.zeros((1, 3), dtype=int)}
        def decode(self, ids, skip_special_tokens=True):
            return 'ok: {"action":"clarify","question":"which person?"}'

    class Model:
        def generate(self, **kw):
            return np.zeros((1, 3 + 5), dtype=int)

    class T:  # minimal torch stand-in
        @staticmethod
        def no_grad():
            import contextlib; return contextlib.nullcontext()

    b = TransformersBackend.__new__(TransformersBackend)
    b.torch, b.model, b.tok, b.device, b.max_new_tokens = T(), Model(), Tok(), "cpu", 10
    out = b.complete([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}], {})
    assert out == '{"action":"clarify","question":"which person?"}'


def test_inline_script_saves_the_first_round_trip_and_reports_latency(episode_path):
    from vi.agent import FakeBackend, ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    fast = ask(engine, "Who took the bike keys?", ep, FakeBackend(), inline_script=True)
    slow = ask(engine, "Who took the bike keys?", ep, FakeBackend(), inline_script=False)
    assert [t["tool"] for t in fast["trace"] if "tool" in t] == ["search_events"]
    assert [t["tool"] for t in slow["trace"] if "tool" in t] == ["get_script", "search_events"]
    assert fast["latency"]["turns"] < slow["latency"]["turns"] and fast["latency"]["total_ms"] >= 0
    assert "SCENE SCRIPT" not in fast["final"]["text"] and fast["final"]["cited"]


@pytest.mark.edge("E-AGT-04")
def test_unknown_event_type_is_rejected_with_the_valid_list():
    from vi.agent.loop import ToolStep, run_tool
    engine = connect()
    with pytest.raises(ValueError, match="pickup"):
        run_tool(engine, ToolStep(tool="search_events", args={"event_type": "pick_up"}))


def test_script_separates_confirmed_people_from_brief_sightings(tmp_path, episode_path):
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    t = Tube(tube_id="c1:0:9", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0),
             last_seen=CamTime(cam_utc_ms=300), box=Box(x1=1, y1=100, x2=30, y2=140))
    grade_tube(t, 1280, 720)
    assert t.quality == "low" and "life 0.3s" in t.quality_reason and "at frame border" in t.quality_reason
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]
    assert "confirmed entities" in get_script(engine, ep, cache=False)
