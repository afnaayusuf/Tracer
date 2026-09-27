"""HTTP API for the engine: serves the static frontend, answers questions over the live store,
exposes footage state, keyframes and the live frame, and starts/stops the ingest as a paced
live stream. One process; the agent model loads once at startup.

  uvicorn vi.api.server:app --host 0.0.0.0 --port 8000
  env: VI_DB, VI_BACKEND (transformers|fake|openai), VI_MODEL, VI_TZ, VI_WEB (static dir), VI_KEYFRAMES, VI_LIVE
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vi.agent import ask_window, clip, episodes_in, footage_bounds, search_events
from vi.store import connect

GREETING = ("hi", "hey", "hello", "yo", "hii", "helo")


class AskIn(BaseModel):
    question: str
    mode: str | None = None
    history: list[dict] | None = None      # previous exchanges [{"q","a","action"}], most recent last


class IngestIn(BaseModel):
    source: str
    start_time: str | None = None
    realtime: bool = True
    profile: str = "common"
    writer: str = "none"
    reid: str = "siglip"
    model: str = "nano"
    fps: float = 4.0
    episode_min: float = 10.0
    grid: str | None = "auto"        # "auto" detects an NVR multiplex layout from the seams; "2x2"/"4x4" to force; null = single camera
    max_width: int = 0


def create_app(db_url: str | None = None, backend_name: str | None = None, model: str | None = None,
               tz_name: str | None = None, web_dir: str | None = None, keyframes_dir: str | None = None,
               live_dir: str | None = None, load_backend: bool = True) -> FastAPI:
    db_url = db_url or os.environ.get("VI_DB", "sqlite+pysqlite:///data/vi.db")
    backend_name = backend_name or os.environ.get("VI_BACKEND", "transformers")
    model = model or os.environ.get("VI_MODEL", "Qwen/Qwen3.5-4B")
    tz_name = tz_name or os.environ.get("VI_TZ", "UTC")
    web_dir = Path(web_dir or os.environ.get("VI_WEB", "ui/web"))
    keyframes_dir = Path(keyframes_dir or os.environ.get("VI_KEYFRAMES", "data/keyframes"))
    live_dir = Path(live_dir or os.environ.get("VI_LIVE", "data/live"))
    tz = ZoneInfo(tz_name)
    engine = connect(db_url)
    app = FastAPI(title="vi-engine")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    state = {"backend": None, "ingest": None, "started": time.time(), "model": model, "backend_name": backend_name}

    def backend():
        if state["backend"] is None:
            if not load_backend or backend_name == "fake":
                from vi.agent import FakeBackend
                state["backend"] = FakeBackend()
            elif backend_name == "openai":
                from vi.agent import OpenAIBackend
                state["backend"] = OpenAIBackend(model=model)
            else:
                from vi.agent import TransformersBackend
                state["backend"] = TransformersBackend(model_id=model)
        return state["backend"]

    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")

    def ingest_status() -> dict:
        p = state["ingest"]
        running = p is not None and p.poll() is None
        st = {}
        try:
            st = json.loads((live_dir / "status.json").read_text())
        except Exception:
            pass
        return {"running": running, "pid": p.pid if running else None, "exit_code": (p.returncode if p is not None and not running else None), **st}

    @app.get("/health")
    def health():
        b = footage_bounds(engine)
        eps = episodes_in(engine, b[0], b[1]) if b else []
        return {"ok": True, "backend": backend_name, "model": model, "tz": tz_name, "uptime_s": round(time.time() - state["started"]),
                "footage": {"start": clock(b[0]), "end": clock(b[1]), "start_ms": b[0], "end_ms": b[1]} if b else None,
                "episodes": len(eps), "ingest": ingest_status()}

    @app.post("/ask")
    def ask(inp: AskIn):
        q = inp.question.strip()
        if not q:
            raise HTTPException(400, "empty question")
        if q.lower().rstrip("!?. ") in GREETING or q.lower().startswith(("hi ", "hey ", "hello ")):
            return {"text": "Hey — ask me about the cameras: who was where, when, what they were carrying, how many people, or what happened in a time window.",
                    "mood": "bot", "evidence": [], "grounding": "greeting", "latency_ms": 0}
        b = footage_bounds(engine)
        now_ms = b[1] if b else int(time.time() * 1000)
        import concurrent.futures
        deadline = float(os.environ.get("VI_ASK_DEADLINE_S", "80"))     # under Cloudflare's 100 s origin limit
        pool = state.setdefault("pool", concurrent.futures.ThreadPoolExecutor(max_workers=1))
        fut = pool.submit(ask_window, engine, q, backend(), now_ms, tz_name, 6, (inp.history or [])[-3:])
        try:
            r = fut.result(timeout=deadline)
        except concurrent.futures.TimeoutError:
            return {"text": f"That took longer than {int(deadline)} s, probably because the engine is busy ingesting. "
                            "Ask a narrower question (a camera or a time window), or try again in a moment.",
                    "mood": "botUnsure", "citations": [], "evidence": [], "grounding": "timeout", "latency_ms": int(deadline * 1000)}
        f = r["final"]
        text = f.get("text") or f.get("question") or ""
        unsure = f.get("action") == "clarify" or f.get("handled_by") in ("scope", "time") or not f.get("cited", False)
        evidence = []
        for c in f.get("citations", [])[:6]:
            if ":E" not in c and "anon:" not in c:
                continue
            try:
                ev = clip(engine, entity_id=c)
            except Exception:
                continue
            for ref in ev.get("keyframe_refs", [])[:1]:
                rel = ref.replace("kf://", "")
                if (keyframes_dir / rel).exists():
                    evidence.append({"entity_id": c, "url": f"/keyframes/{rel}", "label": c.split(":")[-1]})
        return {"text": text, "mood": "botUnsure" if unsure else "bot", "citations": f.get("citations", []), "evidence": evidence,
                "window": r.get("window"), "grounding": r.get("grounding"), "latency_ms": r.get("latency", {}).get("total_ms"),
                "action": f.get("action", "answer"), "trace": r.get("trace", [])}

    @app.get("/episodes")
    def episodes():
        b = footage_bounds(engine)
        if not b:
            return []
        return [{"episode_id": e["episode_id"], "start": clock(e["t0_ms"]), "end": clock(e["t1_ms"] or e["t0_ms"]), "status": e["status"], "tile": e["tile_id"]}
                for e in episodes_in(engine, b[0], b[1])]

    @app.get("/events")
    def events(t_start_ms: int | None = None, t_end_ms: int | None = None, limit: int = 200):
        b = footage_bounds(engine)
        if not b:
            return []
        rows = search_events(engine, t_start_ms=t_start_ms or b[0], t_end_ms=t_end_ms or b[1], limit=limit)
        return [{"t": clock(e["t_ms"]), "type": e["type"], "zone": e["zone_id"], "subjects": e["subject_tube_ids"], "id": e["event_id"]} for e in rows]

    @app.get("/keyframes/{path:path}")
    def keyframe(path: str):
        p = (keyframes_dir / path).resolve()
        if not str(p).startswith(str(keyframes_dir.resolve())) or not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p))

    @app.get("/live/latest.jpg")
    def live_frame():
        p = live_dir / "latest.jpg"
        if not p.exists():
            raise HTTPException(404, "no live frame yet")
        return FileResponse(str(p), headers={"Cache-Control": "no-store"})

    @app.post("/ingest/start")
    def ingest_start(inp: IngestIn):
        if state["ingest"] is not None and state["ingest"].poll() is None:
            return {"started": False, "reason": "already running", **ingest_status()}
        cmd = [sys.executable, "bench/run_ingest.py", "--source", inp.source, "--db", db_url, "--profile", inp.profile, "--reid", inp.reid,
               "--writer", inp.writer, "--model", inp.model, "--fps", str(inp.fps), "--episode-min", str(inp.episode_min), "--live-dir", str(live_dir)]
        if inp.start_time:
            cmd += ["--start-time", inp.start_time]
        if inp.grid:
            cmd += ["--grid", inp.grid]
        if inp.max_width:
            cmd += ["--max-width", str(inp.max_width)]
        if inp.realtime:
            cmd.append("--realtime")
        live_dir.mkdir(parents=True, exist_ok=True)
        log = open(live_dir / "ingest.log", "a")
        state["ingest"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
        return {"started": True, "pid": state["ingest"].pid, "cmd": " ".join(cmd)}

    @app.post("/ingest/stop")
    def ingest_stop():
        p = state["ingest"]
        if p is not None and p.poll() is None:
            p.terminate()
            return {"stopped": True}
        return {"stopped": False, "reason": "not running"}

    @app.get("/ingest/status")
    def ingest_state():
        return ingest_status()

    tiles_path = Path(os.environ.get("VI_TILES", "data/tiles.json"))

    @app.get("/tiles")
    def tiles():
        from vi.fusion import TileMap
        tm = TileMap.load(tiles_path)
        return {"path": str(tiles_path), "tiles": tm.tiles if tm else None, "affinity": tm.affinity if tm else None}

    @app.post("/tiles/recompute")
    def tiles_recompute():
        """Learn which cameras see the same area from the entities recorded so far, and save the map.
        The next ingest start uses it (one identity per tile instead of per camera)."""
        from vi.fusion import discover_tiles
        from vi.fusion.tiles import entities_for_affinity
        from vi.agent.tools import cameras_in_store
        ents = entities_for_affinity(engine)
        tm = discover_tiles(ents, cameras=cameras_in_store(engine))
        tm.save(tiles_path)
        return {"tiles": tm.tiles, "affinity": tm.affinity, "entities_used": len(ents), "saved": str(tiles_path),
                "note": "restart the ingest (POST /ingest/stop then /ingest/start) to apply"}

    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app(load_backend=os.environ.get("VI_LAZY_BACKEND", "1") != "0")
