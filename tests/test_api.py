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
