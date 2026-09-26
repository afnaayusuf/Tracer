"""Block 2 dry run: load episode files into the store, print the scene script, answer the three
canned tool calls a question would need. No LLM yet; this is the evidence layer the agent reads.

  python bench/agent_replay.py --db sqlite+pysqlite:///data/vi.db data/episodes/*.jsonl
  python bench/agent_replay.py --db postgresql+psycopg://vi:vi@localhost/vi data/episodes/*.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, clip, get_script, search_entities, search_events, search_tubes
from vi.store import connect, load_episode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--db", default="sqlite+pysqlite:///data/vi.db")
    ap.add_argument("--script-lines", type=int, default=40)
    ap.add_argument("--ask", action="append", default=[], help="question(s) to run through the agent loop")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--no-inline-script", action="store_true", help="A/B: fetch the script through a tool turn instead")
    a = ap.parse_args()
    engine = connect(a.db)
    loaded = [load_episode_file(engine, p) for p in a.paths]
    ep = loaded[-1]["episode_id"]
    script = get_script(engine, ep, cache=False)
    print("\n".join(script.splitlines()[: a.script_lines]))
    if len(script.splitlines()) > a.script_lines:
        print(f"  ... ({len(script.splitlines()) - a.script_lines} more lines)")
    ents = search_entities(engine, class_label="person")
    enters = search_events(engine, event_type="enter_zone")
    relinks = search_events(engine, event_type="relink")
    first = ents[0]["entity_id"] if ents else None
    evidence = clip(engine, entity_id=first) if first else {}
    row = {"ring": "agent", "db": a.db.split("://")[0], "episodes_loaded": len(loaded), "episode_id": ep,
           "rows": {k: sum(l.get(k, 0) for l in loaded) for k in ("ticks", "events", "tubes", "entities", "patches", "custody")},
           "person_entities": len(ents), "enter_events": len(enters), "relink_events": len(relinks),
           "script_chars": len(script), "script_lines": len(script.splitlines()),
           "clip_first_entity": {"tubes": len(evidence.get("tube_ids", [])), "keyframes": len(evidence.get("keyframe_refs", []))},
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    answers = []
    if a.ask:
        if a.backend == "fake":
            backend = FakeBackend()
        elif a.backend == "openai":
            backend = OpenAIBackend(base_url=a.base_url, model=a.model)
        else:
            try:
                backend = TransformersBackend(model_id=a.model)
                print(f"[agent] transformers backend loaded {a.model} via {backend.loader}")
            except Exception as e:
                print(f"[agent] transformers backend failed ({type(e).__name__}: {str(e)[:120]}); using fake backend")
                backend = FakeBackend()
        for q in a.ask:
            res = ask(engine, q, ep, backend, inline_script=not a.no_inline_script)
            answers.append(res)
            calls = [f"{t['tool']}({', '.join(f'{k}={v}' for k, v in t['args'].items())}) -> "
                     + (f"ERROR {t['error'][:80]}" if t.get('error') else str(t['results'])) for t in res["trace"] if "tool" in t]
            lat = res["latency"]
            print(f"\nQ: {q}\n   latency: {lat['total_ms'] / 1000:.1f}s total (model {lat['model_ms'] / 1000:.1f}s, tools {lat['tool_ms']}ms, "
                  f"{lat['turns']} turn(s), first prompt {lat['first_prompt_chars']} chars)\n   tools: " + (" | ".join(calls) or "none"))
            f = res["final"]
            if f["action"] == "answer":
                print(f"   A ({f.get('confidence', 0):.2f}{'' if f['cited'] else ', UNCITED'}): {f['text']}\n   cites: {', '.join(f['citations']) or '—'}")
            else:
                print(f"   clarify: {f['question']}")
        row["answers"] = [{"q": r["question"], "steps": r["steps"], "cited": r["final"].get("cited"),
                           "action": r["final"]["action"], "backend": r["backend"], **r["latency"]} for r in answers]
        row["latency_p50_ms"] = int(sorted(r["latency"]["total_ms"] for r in answers)[len(answers) // 2])
        row["latency_max_ms"] = max(r["latency"]["total_ms"] for r in answers)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "agent.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "answers"}, indent=2))


if __name__ == "__main__":
    main()
