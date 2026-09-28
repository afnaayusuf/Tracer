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
import threading
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


class DescribeIn(BaseModel):
    camera_id: str
    tube_ids: list[str]
    crops_jpeg_b64: list[str]              # one JPEG per tube, base64
    mode: str = "appearance"               # appearance | activity | scene (one full frame -> inventory) | narrate (frames of one person -> steps)


class VolumeIn(BaseModel):
    camera_id: str                          # camera or tile id (composite view)
    t0_ms: int
    period_ms: int
    times_ms: list[int]
    frames_jpeg_b64: list[str]              # annotated frames, time order
    ids_present: list[str] = []
    previous_summary: str | None = None
    state: dict | None = None               # stream mode: the current state; the reply is a delta


class InspectIn(BaseModel):
    question: str
    entity_id: str | None = None
    camera_id: str | None = None
    t_ms: int | None = None


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
    tiles: str = "auto"              # "one" = homo BuF (all cameras one space); "auto" = learned map if present; "none" = per camera
    auto_tiles_after_min: float = 5  # with tiles="auto" and no map yet: learn the map after this much footage and restart seamlessly
    skip_s: float = 0
    volume_s: float = 8.0            # generator period; stream mode uses 1 s
    stream: bool = False             # stream mode: 3 fps, one delta per second, immediate lib write


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
    state = {"backend": None, "ingest": None, "started": time.time(), "model": model, "backend_name": backend_name,
             "writer": None, "model_lock": threading.Lock(), "questions_waiting": 0, "sheets": 0, "sheet_ms": [], "restarts": 0}
    tiles_path = Path(os.environ.get("VI_TILES") or (live_dir / "tiles.json"))
    os.environ["VI_TILES"] = str(tiles_path)                      # the feed side reads the same map

    state["load_lock"] = threading.Lock()

    def backend():
        """One copy of the model per process. The lock matters: the ingest's first /describe and the
        first /ask arrive together, and two loads of a 4B do not fit on an L4 (session 40)."""
        if state["backend"] is not None:
            return state["backend"]
        with state["load_lock"]:
            if state["backend"] is None:
                if not load_backend or backend_name == "fake":
                    from vi.agent import FakeBackend
                    state["backend"] = FakeBackend()
                elif backend_name == "openai":
                    from vi.agent import OpenAIBackend
                    state["backend"] = OpenAIBackend(model=model, base_url=os.environ.get("VI_OPENAI_BASE", "http://127.0.0.1:8001/v1"))
                else:
                    from vi.agent import TransformersBackend
                    state["backend"] = TransformersBackend(model_id=model)
        return state["backend"]

    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%H:%M:%S")

    def writer():
        """The writer shares the agent's loaded model: one copy of the weights on the GPU, one process."""
        if state["writer"] is None:
            be = backend()
            if getattr(be, "name", "") == "transformers" and getattr(be, "model", None) is not None:
                from vi.writer import WriterVLM
                state["writer"] = WriterVLM(backend=be)
            elif getattr(be, "name", "") == "openai":
                from vi.writer import OpenAIWriter
                state["writer"] = OpenAIWriter(base_url=be.base_url, model=be.model)     # the vLLM server: images as data URLs
            else:
                state["writer"] = "fake"
        return state["writer"]

    @app.post("/describe")
    def describe(inp: DescribeIn):
        """Contact-sheet descriptions for the ingest. Questions have priority: a sheet waits while a
        question is in flight, and the ingest never blocks on this call (it posts and moves on)."""
        import base64, io
        import numpy as np
        from PIL import Image
        crops = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")) for b in inp.crops_jpeg_b64]
        w = writer()
        if w == "fake":
            if inp.mode == "scene":
                return {"scene": [{"object": "keyboard", "where": "on the desk", "state": None, "count": None, "confidence": 0.8},
                                  {"object": "cardboard box", "where": "on the counter", "state": "open", "count": 2, "confidence": 0.8}], "sheet_ms": 0}
            if inp.mode == "narrate":
                return {"narration": {"steps": ["takes a small box out of the big box", "inspects a glass jug", "puts it back"],
                                      "objects_handled": [{"object": "small box", "state": "opened"}, {"object": "glass jug", "state": "inspected"}],
                                      "summary": "He unpacks small boxes of glass jugs and checks each one.", "counts": {"small box": 12}, "confidence": 0.7}, "sheet_ms": 0}
            cells = [{"tube_id": t, "attributes": {"modality": "rgb", "description": "person", "top_color": "orange", "confidence": 0.7,
                                                   **({"activity": "packing boxes", "objects_nearby": ["box"], "attention": "down", "posture": "standing"} if inp.mode == "activity" else {})}}
                     for t in inp.tube_ids]
            return {"cells": cells, "sheet_ms": 0}
        for _ in range(200):                                     # yield to questions (max ~20 s of waiting)
            if state["questions_waiting"] == 0:
                break
            time.sleep(0.1)
        with state["model_lock"]:
            if inp.mode == "scene":
                items = w.scene(crops[0]) if crops else []
                state["sheets"] += 1; state["sheet_ms"].append(w.last_ms); state["sheet_ms"] = state["sheet_ms"][-50:]
                return {"scene": items, "sheet_ms": w.last_ms}
            if inp.mode == "narrate":
                nar = w.narrate(crops)
                state["sheets"] += 1; state["sheet_ms"].append(w.last_ms); state["sheet_ms"] = state["sheet_ms"][-50:]
                return {"narration": nar, "sheet_ms": w.last_ms}
            res = w.describe(crops, inp.tube_ids, mode=inp.mode)
        state["sheets"] += 1; state["sheet_ms"].append(w.last_ms); state["sheet_ms"] = state["sheet_ms"][-50:]
        if res is None:
            return {"cells": [], "sheet_ms": w.last_ms}
        return {"cells": [{"tube_id": c.tube_id, "attributes": c.attributes.model_dump(mode="json")} for c in res.cells], "sheet_ms": w.last_ms}

    def inspect_fn(question: str, entity_id: str | None = None, camera_id: str | None = None, t_ms: int | None = None) -> dict:
        """Look at pixels for a question the lib cannot answer: the entity's keyframe nearest t (or its
        latest), or the live frame of a camera. Slow (one VLM call); the answer cites the frame."""
        import numpy as np
        from PIL import Image
        ref, img = None, None
        if entity_id:
            try:
                ev = clip(engine, entity_id=entity_id)
                refs = ev.get("keyframe_refs", [])
                if refs:
                    ref = refs[-1]
                    p = keyframes_dir / ref.replace("kf://", "")
                    if p.exists():
                        img = np.asarray(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if img is None:
            p = live_dir / "latest.jpg"
            if p.exists():
                img = np.asarray(Image.open(p).convert("RGB")); ref = "live frame"
        if img is None:
            return {"answer": "No frame is available to look at yet.", "ref": None}
        w = writer()
        if w == "fake":
            return {"answer": f"(inspected {ref}) a person at a counter", "ref": ref}
        with state["model_lock"]:
            text = w.inspect(img, question)
        return {"answer": text, "ref": ref, "ms": w.last_ms}

    def inspect_fn_unlocked(question: str, entity_id: str | None = None, camera_id: str | None = None, t_ms: int | None = None) -> dict:
        """inspect() for use INSIDE ask (which already holds the model lock). Never raises: a failure is an answer."""
        import numpy as np
        from PIL import Image
        try:
            ref, img = None, None
            if entity_id:
                try:
                    refs = clip(engine, entity_id=entity_id).get("keyframe_refs", [])
                    if refs:
                        ref = refs[-1]; p = keyframes_dir / ref.replace("kf://", "")
                        if p.exists(): img = np.asarray(Image.open(p).convert("RGB"))
                except Exception:
                    pass
            if img is None and (live_dir / "latest.jpg").exists():
                img = np.asarray(Image.open(live_dir / "latest.jpg").convert("RGB")); ref = "live frame"
            if img is None:
                return {"answer": "No frame is available to look at yet.", "ref": None}
            w = writer()
            if w == "fake":
                return {"answer": f"(inspected {ref}) a person at a counter", "ref": ref}
            return {"answer": w.inspect(img, question), "ref": ref, "ms": w.last_ms}
        except Exception as e:
            import traceback
            print(f"[inspect] error: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            return {"answer": f"inspect failed ({type(e).__name__}: {str(e)[:120]}); answer from the script instead.", "ref": None}

    @app.post("/volume")
    def volume(inp: VolumeIn):
        """The generator's call: one frame-volume in, one structured record out. Questions take priority."""
        import base64, io
        import numpy as np
        from PIL import Image
        from vi.writer import FrameVolume, build_prompt
        frames = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")) for b in inp.frames_jpeg_b64]
        vol = FrameVolume(camera_id=inp.camera_id, t0_ms=inp.t0_ms, period_ms=inp.period_ms, frames=frames, times_ms=inp.times_ms, ids_present=inp.ids_present)
        prompt = build_prompt(len(frames), inp.period_ms / 1000, inp.previous_summary, inp.ids_present)
        w = writer()
        if w == "fake":
            return {"record": {"people": [{"id": (inp.ids_present or ["unlabeled"])[0], "appearance": "person", "actions": ["moves through the view"],
                                           "objects_handled": [], "attention": None, "posture": "walking", "location": "centre"}],
                               "objects": [{"object": "counter", "where": "centre", "state": None, "count": None, "changed": False}],
                               "events": [], "summary": "One person moves through the view.", "unclear": [], "confidence": 0.6}, "ms": 0, "video_input": False}
        for _ in range(200):
            if state["questions_waiting"] == 0:
                break
            time.sleep(0.1)
        with state["model_lock"]:
            rec = w.volume(vol, prompt, tz_name)
        state["sheets"] += 1; state["sheet_ms"].append(w.last_ms); state["sheet_ms"] = state["sheet_ms"][-50:]
        return {"record": rec, "ms": w.last_ms, "video_input": getattr(w, "video_ok", None), "error": getattr(w, "last_error", None) if rec is None else None}

    @app.post("/delta")
    def delta(inp: VolumeIn):
        """Stream mode: the last second's frames + the state -> only what changed (<= 80 tokens). Questions still first."""
        import base64, io
        import numpy as np
        from PIL import Image
        from vi.writer import FrameVolume
        frames = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")) for b in inp.frames_jpeg_b64]
        vol = FrameVolume(camera_id=inp.camera_id, t0_ms=inp.t0_ms, period_ms=inp.period_ms, frames=frames, times_ms=inp.times_ms, ids_present=inp.ids_present)
        w = writer()
        if w == "fake":
            pid = (inp.ids_present or ["unlabeled"])[0]
            return {"delta": {"people": {pid: {"action": "moves through the view", "objects": []}}, "objects": [], "event": None, "empty": False}, "ms": 0}
        for _ in range(100):
            if state["questions_waiting"] == 0:
                break
            time.sleep(0.05)
        with state["model_lock"]:
            d = w.delta(vol, inp.state or {}, tz_name)
        state["sheets"] += 1; state["sheet_ms"].append(w.last_ms); state["sheet_ms"] = state["sheet_ms"][-50:]
        return {"delta": d, "ms": w.last_ms, "video_input": getattr(w, "video_ok", None)}

    @app.post("/inspect")
    def inspect(inp: InspectIn):
        return inspect_fn(inp.question, inp.entity_id, inp.camera_id, inp.t_ms)

    def ingest_status() -> dict:
        p = state["ingest"]
        running = p is not None and p.poll() is None
        st = {}
        try:
            st = json.loads((live_dir / "status.json").read_text())
        except Exception:
            pass
        return {"running": running, "pid": p.pid if running else None, "exit_code": (p.returncode if p is not None and not running else None),
                "restarts": state["restarts"], "writer_sheets": state["sheets"],
                "writer_ms_p50": (sorted(state["sheet_ms"])[len(state["sheet_ms"]) // 2] if state["sheet_ms"] else None), **st}

    def status_answer() -> dict:
        st = ingest_status()
        b = footage_bounds(engine)
        bits = []
        if st.get("running"):
            bits.append(f"Yes. The engine is processing (pid {st['pid']}): {st.get('cameras') or '?'} camera(s), "
                        f"{st.get('frames') or 0} frames so far, footage position {clock(st['now_ms']) if st.get('now_ms') else '?'}"
                        + (f", keeping up at {st['footage_s'] / max(st['wall_s'], 1):.2f}x real time" if st.get('footage_s') and st.get('wall_s') else "")
                        + f", {st.get('episodes', 0)} episode(s) in the lib" + (f", {st.get('sheets', 0)} description sheets" if st.get('sheets') else "") + ".")
        else:
            bits.append("No ingest is running right now" + (f" (the last one exited with code {st['exit_code']})" if st.get("exit_code") is not None else "") + ".")
        if b:
            bits.append(f"The lib covers {clock(b[0])}–{clock(b[1])}" + (f" (updated {max(0, (st['now_ms'] - b[1]) // 1000)} s of footage ago)." if st.get("now_ms") else "."))
        else:
            bits.append("The lib is empty so far (the first flush comes about 10 s after the first person is seen).")
        try:
            tail = [l for l in (live_dir / "ingest.log").read_text().splitlines() if l.startswith("[episode]") or "Error" in l or "Traceback" in l][-1:]
            if tail:
                bits.append("Last log line: " + tail[0][:140])
        except Exception:
            pass
        return {"text": " ".join(bits), "mood": "bot" if st.get("running") else "botUnsure", "citations": [], "evidence": [],
                "grounding": "status", "latency_ms": 0, "action": "answer"}

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
        from vi.agent import classify_scope
        if classify_scope(q)[0] == "status":
            return status_answer()
        b = footage_bounds(engine)
        now_ms = b[1] if b else int(time.time() * 1000)
        import concurrent.futures
        deadline = float(os.environ.get("VI_ASK_DEADLINE_S", "80"))     # under Cloudflare's 100 s origin limit
        pool = state.setdefault("pool", concurrent.futures.ThreadPoolExecutor(max_workers=2))
        def _run():
            state["questions_waiting"] += 1
            try:
                with state["model_lock"]:                            # the writer yields; one model, questions first
                    steps = int(os.environ.get("VI_MAX_STEPS", "4" if backend_name == "transformers" else "6"))
                    return ask_window(engine, q, backend(), now_ms, tz_name, steps, (inp.history or [])[-3:], inspector=inspect_fn_unlocked)
            finally:
                state["questions_waiting"] -= 1
        fut = pool.submit(_run)
        try:
            r = fut.result(timeout=deadline)
        except concurrent.futures.TimeoutError:
            return {"text": f"That took longer than {int(deadline)} s. Ask a narrower question (a camera or a time window), or try again in a moment.",
                    "mood": "botUnsure", "citations": [], "evidence": [], "grounding": "timeout", "latency_ms": int(deadline * 1000), "action": "answer"}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ask] error: {type(e).__name__}: {e}\n{tb}", flush=True)
            try:
                (live_dir / "ask_errors.log").open("a").write(f"{datetime.now().isoformat()} {q!r}\n{tb}\n")
            except Exception:
                pass
            return {"text": f"The engine hit an error answering that ({type(e).__name__}: {str(e)[:160]}). It has been logged; try rephrasing.",
                    "mood": "botUnsure", "citations": [], "evidence": [], "grounding": "error", "latency_ms": 0, "action": "answer",
                    "error": f"{type(e).__name__}: {str(e)[:300]}"}
        f = r["final"]
        text = f.get("text") or f.get("question") or ""
        unsure = f.get("action") == "clarify" or f.get("handled_by") in ("scope", "time") or not f.get("cited", False)
        evidence = []
        cites = list(f.get("citations", []))
        if any(c.startswith("W") and c[1:].isdigit() for c in cites) and r.get("window_ms"):
            from vi.agent.tools import world_members
            members = world_members(engine, r["window_ms"][0], r["window_ms"][1], r.get("camera_id"))
            for c in list(cites):
                if c in members:
                    cites += members[c]
        for c in cites[:12]:
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
        writer_mode = inp.writer
        if writer_mode in ("qwen", "auto"):
            writer_mode = "remote"                                   # the API's own model describes; the ingest never loads a VLM
        cmd = [sys.executable, "bench/run_ingest.py", "--source", inp.source, "--db", db_url, "--profile", inp.profile, "--reid", inp.reid,
               "--writer", writer_mode, "--writer-url", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/describe",
               "--model", inp.model, "--fps", str(inp.fps), "--episode-min", str(inp.episode_min), "--live-dir", str(live_dir),
               "--tiles", inp.tiles, "--volume-s", str(inp.volume_s)] + (["--stream"] if inp.stream else [])
        if inp.skip_s:
            cmd += ["--skip-s", str(inp.skip_s)]
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
        state["ingest_params"] = inp
        threading.Thread(target=_watch_ingest, args=(state["ingest"], inp), daemon=True).start()
        if inp.tiles == "auto" and not tiles_path.exists() and inp.auto_tiles_after_min > 0:
            threading.Thread(target=_auto_tiles, args=(inp, state["ingest"]), daemon=True).start()
        return {"started": True, "pid": state["ingest"].pid, "cmd": " ".join(cmd), "auto_tiles": inp.tiles == "auto" and not tiles_path.exists()}

    def _watch_ingest(proc, inp: IngestIn) -> None:
        """If the ingest dies with an error, restart it from the footage position it reached (up to 3 times)."""
        rc = proc.wait()
        if rc == 0 or state["ingest"] is not proc:
            return
        if state["restarts"] >= 3:
            return
        try:
            st = json.loads((live_dir / "status.json").read_text()); pos = float(st.get("footage_s", 0))
        except Exception:
            pos = 0.0
        state["restarts"] += 1
        time.sleep(5)
        ingest_start(inp.model_copy(update={"skip_s": pos, "auto_tiles_after_min": 0}))

    def _auto_tiles(inp: IngestIn, proc) -> None:
        """Phase 2 without anyone typing: after N minutes of footage, learn the map; if cameras group
        (homo or hetero), restart the ingest with the map from where it got to."""
        from vi.fusion import discover_tiles
        from vi.fusion.tiles import entities_for_affinity
        from vi.agent.tools import cameras_in_store
        deadline = time.time() + 3 * 3600
        while time.time() < deadline and proc.poll() is None:
            time.sleep(15)
            try:
                st = json.loads((live_dir / "status.json").read_text())
            except Exception:
                continue
            if st.get("footage_s", 0) < inp.auto_tiles_after_min * 60:
                continue
            ents = entities_for_affinity(engine)
            tm = discover_tiles(ents, cameras=cameras_in_store(engine))
            grouped = any(len(c) > 1 for c in tm.tiles.values())
            tm.save(tiles_path)
            (live_dir / "auto_tiles.json").write_text(json.dumps({"at_footage_s": st.get("footage_s"), "tiles": tm.tiles, "adjacency": tm.adjacency,
                                                                 "kind": tm.kind, "entities_used": len(ents), "restarted": grouped}))
            if grouped and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    pass
                inp2 = inp.model_copy(update={"skip_s": float(st.get("footage_s", 0)), "tiles": "auto", "auto_tiles_after_min": 0})
                ingest_start(inp2)
            return

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

    # ---------------------------------------------------------------- the lib, readable directly
    def lib_summary(window_min: float | None = None, since_ms: int | None = None) -> dict:
        from vi.agent.tools import activities_window, coverage_window, hard_evidence_people, load_tile_map, scene_inventory, volumes_window, world_groups
        b = footage_bounds(engine)
        st = ingest_status()
        if b is None:
            return {"footage": None, "people": [], "activities": [], "events": [], "episodes": [], "ingest": st, "now_ms": None}
        t0, t1 = b
        ws = max(t0, t1 - int(window_min * 60_000)) if window_min else t0
        cast = coverage_window(engine, ws, t1)
        worlds = world_groups(cast, tile_map=load_tile_map())
        acts = activities_window(engine, ws, t1)
        last_act: dict[str, dict] = {}
        for x in acts:
            if x.get("entity_id"):
                last_act[worlds.get(x["entity_id"], x["entity_id"])] = x
        people = {}
        for c in cast:
            w = worlds[c["entity_id"]]
            pr = people.setdefault(w, {"id": w, "first_seen_ms": c["first_seen_ms"], "last_seen_ms": c["last_seen_ms"], "cameras": set(), "tracks": [],
                                       "looks": None, "keyframes": [], "coverage": 0.0, "quality": "ok"})
            pr["first_seen_ms"] = min(pr["first_seen_ms"], c["first_seen_ms"]); pr["last_seen_ms"] = max(pr["last_seen_ms"], c["last_seen_ms"])
            pr["cameras"] |= set(c["cameras"]); pr["tracks"].append(c["entity_id"]); pr["coverage"] = max(pr["coverage"], c["coverage"])
            if c.get("looks") and not str(c["looks"]).startswith("NOT A PERSON") and (pr["looks"] is None or len(str(c["looks"])) > len(str(pr["looks"]))):
                pr["looks"] = c["looks"]
            if c.get("keyframe"):
                rel = c["keyframe"].replace("kf://", "")
                if (keyframes_dir / rel).exists() and len(pr["keyframes"]) < 4:
                    pr["keyframes"].append(f"/keyframes/{rel}")
        out_people = []
        for w, pr in sorted(people.items(), key=lambda kv: kv[1]["first_seen_ms"]):
            la = last_act.get(w)
            out_people.append({**pr, "cameras": sorted(pr["cameras"]), "first_seen": clock(pr["first_seen_ms"]), "last_seen": clock(pr["last_seen_ms"]),
                               "live": (t1 - pr["last_seen_ms"]) < 15_000,
                               "last_activity": ({"t": clock(la["t_ms"]), "activity": la["activity"], "objects_nearby": la["objects_nearby"],
                                                  "attention": la["attention"], "posture": la["posture"]} if la else None)})
        evs = search_events(engine, t_start_ms=(since_ms or ws), t_end_ms=t1, limit=300)
        acts_out = [{"t_ms": x["t_ms"], "t": clock(x["t_ms"]), "person": worlds.get(x["entity_id"], x["entity_id"]), "camera": x["camera_id"],
                     "activity": x["activity"], "objects_nearby": x["objects_nearby"], "attention": x["attention"], "posture": x["posture"],
                     "carried_item": x["carried_item"], "kind": x.get("kind"), "steps": x.get("steps") or [], "objects_handled": x.get("objects_handled") or [],
                     "counts": x.get("counts") or {}} for x in acts if since_ms is None or x["t_ms"] > since_ms]
        scene = {cam: {"t": clock(d["t_ms"]), "objects": d["objects"]} for cam, d in scene_inventory(engine, t1).items()}
        vols = volumes_window(engine, ws, t1)
        if vols and vols[-1]["objects"]:
            scene[vols[-1]["camera_id"] or "composite"] = {"t": clock(vols[-1]["t_end_ms"]), "objects": vols[-1]["objects"]}
        periods = [{"t0": clock(v["t0_ms"] or v["t_end_ms"]), "t1": clock(v["t_end_ms"]), "t_end_ms": v["t_end_ms"], "summary": v["summary"], "people": v["people"],
                    "changed": [o for o in v["objects"] if o.get("changed")], "events": v["events"], "confidence": v["confidence"]}
                   for v in vols if since_ms is None or v["t_end_ms"] > since_ms][-60:]
        return {"footage": {"start": clock(t0), "end": clock(t1), "start_ms": t0, "end_ms": t1, "minutes": round((t1 - t0) / 60000, 1)},
                "now_ms": t1, "window": {"start": clock(ws), "end": clock(t1)}, "people": out_people, "people_count": len(out_people),
                "hard_evidence_min_people": hard_evidence_people(cast),
                "activities": acts_out[-200:], "scene": scene, "periods": periods, "events": [{"t_ms": e["t_ms"], "t": clock(e["t_ms"]), "type": e["type"], "camera": e["camera_id"], "zone": e["zone_id"],
                                                           "who": [worlds.get(x, x) for x in (e["subject_entity_ids"] or [])] or (e["subject_tube_ids"] or [])} for e in evs
                                                          if since_ms is None or e["t_ms"] > since_ms][-200:],
                "episodes": [{"id": e["episode_id"][:12], "start": clock(e["t0_ms"]), "end": clock(e["t1_ms"] or e["t0_ms"]), "status": e["status"]}
                             for e in episodes_in(engine, t0, t1)][-30:],
                "ingest": st}

    @app.get("/lib/summary")
    def lib_summary_ep(window_min: float | None = None, since_ms: int | None = None):
        try:
            return lib_summary(window_min, since_ms)
        except Exception as e:
            import traceback
            print(f"[lib] summary error: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            return {"footage": None, "people": [], "activities": [], "events": [], "episodes": [], "ingest": ingest_status(), "now_ms": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    @app.get("/lib/stream")
    def lib_stream(window_min: float = 30.0, max_events: int = 0):
        """Server-sent events: a fresh summary every 2 s while the lib changes (people, activities, events, ingest).
        max_events > 0 bounds the stream (tests, curl)."""
        from fastapi.responses import StreamingResponse
        def gen():
            last_key = None
            sent = 0
            while max_events <= 0 or sent < max_events:
                sent += 1
                try:
                    summ = lib_summary(window_min)
                    key = (summ.get("now_ms"), len(summ.get("activities", [])), len(summ.get("events", [])), summ.get("people_count"),
                           (summ.get("ingest") or {}).get("frames"))
                    if key != last_key:
                        last_key = key
                        yield f"data: {json.dumps(summ)}\n\n"
                    else:
                        yield ": keepalive\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)[:200]})}\n\n"
                if max_events <= 0 or sent < max_events:
                    time.sleep(2)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/deploy/report")
    def deploy_report():
        """Deployability, measured on the running system: does each lane keep up with live footage, and how far
        can it be pushed? Perception lane: decode+detect+track per camera-frame. Generator lane: one model call per
        frame-volume. Both from the ingest's own counters."""
        import math
        st = ingest_status()
        fps = float(st.get("fps") or 4.0); cams = int(st.get("cameras") or 1)
        det = st.get("detect_ms_p50"); foot, wall = st.get("footage_s"), st.get("wall_s")
        rt = round(foot / wall, 2) if foot and wall else None
        # perception: the batch cost is nearly flat in batch size on one GPU (measured: 27 ms for 1 cell, 30 ms for 4, ~166 ms for 16 with medium)
        per_frame_budget_ms = 1000.0 / fps
        headroom = round(per_frame_budget_ms / det, 1) if det else None
        max_cams_est = None
        if det:
            per_cell = det / max(1, cams)                                      # amortised cost per camera-frame in the batch
            max_cams_est = int(per_frame_budget_ms / max(per_cell, 5.0))       # floor of 5 ms/cell: decode + track are not free
        vol_s = st.get("volume_s"); vms = st.get("volume_ms_p50"); cov = st.get("generator_coverage")
        sustainable_period = math.ceil(vms / 1000 * 1.2) if vms else None
        report = {
            "perception": {"cameras": cams, "fps": fps, "detect_ms_p50_per_batch": det, "realtime_factor": rt,
                           "keeps_up": (rt is not None and rt >= 0.97), "batch_headroom_x": headroom,
                           "estimated_max_cameras_at_this_fps": max_cams_est,
                           "note": "realtime_factor is 1.0 when paced to the clock; headroom is the unpaced margin"},
            "generator": {"mode": "stream (deltas)" if st.get("stream") else "periods", "volume_period_s": vol_s, "call_ms_p50": vms, "coverage": cov,
                          "volumes_done": st.get("volumes_done"), "volumes_dropped": st.get("volumes_dropped"), "keeps_up": (cov is not None and cov >= 0.9),
                          "sustainable_period_s": sustainable_period, "frame_to_lib_ms_p50": st.get("frame_to_lib_ms_p50"),
                          "meets_1s": (st.get("frame_to_lib_ms_p50") is not None and st["frame_to_lib_ms_p50"] <= 1000),
                          "note": "coverage = periods with a record / periods elapsed; frame_to_lib = newest frame captured -> its delta in the store"},
            "model": {"name": model, "backend": backend_name, "sheet_ms_p50": (sorted(state["sheet_ms"])[len(state["sheet_ms"]) // 2] if state["sheet_ms"] else None)},
            "gpu": None,
        }
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                report["gpu"] = {"name": torch.cuda.get_device_name(0), "total_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1)}
        except Exception:
            pass
        verdict = []
        if report["perception"]["keeps_up"]: verdict.append(f"perception keeps up at {fps:g} fps x {cams} cameras" + (f" (~{max_cams_est} cameras possible)" if max_cams_est else ""))
        elif rt is not None: verdict.append(f"perception is behind real time ({rt}x)")
        if cov is not None:
            verdict.append(f"generator covers {int(cov * 100)}% of periods at {vol_s}s" + (f"; sustainable period ~{sustainable_period}s on this model/GPU" if sustainable_period else ""))
        if st.get("frame_to_lib_ms_p50") is not None:
            verdict.append(f"frame->lib {st['frame_to_lib_ms_p50'] / 1000:.2f}s p50 " + ("(meets the 1 s target)" if st["frame_to_lib_ms_p50"] <= 1000 else "(above 1 s)"))
        report["verdict"] = "; ".join(verdict) or "not enough footage yet"
        return report

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
