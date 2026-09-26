"""Temporary question interface over the live store. Runs in Colab (share link) or locally.

  python ui/app.py --db "$DB_URL" --backend transformers --model Qwen/Qwen3.5-4B --tz Asia/Kolkata --share

Left: footage status, chat. Right: evidence keyframes for the cited entities, the trace, and the
episode list. Every answer shows its time window, latency and citations; refusals (future time,
off-topic, actions) come from the deterministic guards, not the model.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask_window, clip, episodes_in, footage_bounds
from vi.store import connect


def build(db_url: str, backend_name: str, model: str, tz_name: str, keyframes_dir: str = "data/keyframes"):
    engine = connect(db_url)
    tz = ZoneInfo(tz_name)
    if backend_name == "fake":
        backend = FakeBackend()
    elif backend_name == "openai":
        backend = OpenAIBackend(model=model)
    else:
        backend = TransformersBackend(model_id=model)

    def clock(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%Y-%m-%d %H:%M:%S")

    def status() -> str:
        b = footage_bounds(engine)
        if b is None:
            return "**No footage processed yet.** Start `bench/run_ingest.py` and ask again."
        eps = episodes_in(engine, b[0], b[1])
        return (f"**Footage** {clock(b[0])} → {clock(b[1])} ({tz_name})  ·  **episodes** {len(eps)}  ·  "
                f"**now** {clock(b[1])}  ·  backend `{getattr(backend, 'name', backend_name)}` `{model}`")

    def episode_rows() -> list[list]:
        b = footage_bounds(engine)
        if b is None:
            return []
        return [[e["episode_id"][:12], clock(e["t0_ms"]), clock(e["t1_ms"] or e["t0_ms"]), e["status"], e["tile_id"]] for e in episodes_in(engine, b[0], b[1])][-25:]

    def keyframes_for(citations: list[str]) -> list:
        ims = []
        for c in citations:
            if ":E" not in c and "anon:" not in c:
                continue
            try:
                ev = clip(engine, entity_id=c)
            except Exception:
                continue
            for ref in ev.get("keyframe_refs", [])[:2]:
                p = Path(keyframes_dir) / ref.replace("kf://", "")
                if p.exists():
                    ims.append((str(p), c))
        return ims[:12]

    def ask(question: str, history: list):
        history = history or []
        if not question.strip():
            return history, "", [], {}, status(), episode_rows()
        b = footage_bounds(engine)
        now_ms = b[1] if b else int(time.time() * 1000)
        t0 = time.perf_counter()
        r = ask_window(engine, question, backend, now_ms, tz_name)
        f = r["final"]
        text = f.get("text") or f.get("question") or ""
        meta = []
        if r.get("window"):
            meta.append(f"window {r['window']}")
        meta.append(f"{r['latency']['total_ms'] / 1000:.1f}s")
        if f.get("citations"):
            meta.append("cites " + ", ".join(f["citations"][:8]))
        if f.get("handled_by"):
            meta.append(f"guard: {r.get('grounding')}")
        if f.get("numeric_issue"):
            meta.append("numbers checked")
        answer = text + "\n\n" + " · ".join(f"`{m}`" for m in meta)
        history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        trace = {"grounding": r.get("grounding"), "window": r.get("window"), "steps": r.get("steps"), "trace": r.get("trace", []),
                 "latency": r.get("latency"), "citations": f.get("citations", []), "rejected": f.get("rejected_citations", [])}
        return history, "", keyframes_for(f.get("citations", [])), trace, status(), episode_rows()

    with gr.Blocks(title="vi-engine") as demo:
        gr.Markdown("## vi-engine — ask the footage")
        st = gr.Markdown(status())
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Answers", height=460)
                q = gr.Textbox(label="Question", placeholder="Who was at the conveyor between 10:10 and 10:20? · How many people in the last 15 minutes? · What was the person in the white hat carrying?")
                with gr.Row():
                    btn = gr.Button("Ask", variant="primary")
                    refresh = gr.Button("Refresh status")
            with gr.Column(scale=2):
                gal = gr.Gallery(label="Evidence (keyframes of cited entities)", columns=3, height=300)
                tr = gr.JSON(label="Trace")
                eps = gr.Dataframe(headers=["episode", "start", "end", "status", "tile"], value=episode_rows(), label="Episodes", interactive=False)
        btn.click(ask, [q, chat], [chat, q, gal, tr, st, eps])
        q.submit(ask, [q, chat], [chat, q, gal, tr, st, eps])
        refresh.click(lambda: (status(), episode_rows()), None, [st, eps])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--backend", choices=["fake", "transformers", "openai"], default="transformers")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo = build(a.db, a.backend, a.model, a.tz)
    demo.launch(share=a.share, server_port=a.port, server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
