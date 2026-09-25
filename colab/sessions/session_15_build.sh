#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 15 build: a reasoning path that withstands the runtime.
#   * colab/preflight.sh: prints what the runtime is (python, ensurepip, uv, torch/torchaudio CUDA,
#     GPU, disk) before anything assumes it
#   * colab/vllm_venv.sh: venv built by uv -> venv --without-pip + get-pip -> virtualenv, each failure
#     printed; vLLM installed and served from it; the runtime's torch is never touched
#   * TransformersBackend: in-process Qwen3.5-4B via transformers, no server, no second torch
#   * backend chain for the questions: vLLM (openai) -> transformers -> fake, with the reason at
#     each step; the session cannot end without an answer path
#
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_15.sh
#  Env: GH_REPO (afnaayusuf/Tracer) GH_TOKEN REPO_DIR (/content/Tracer) SOURCE (/content/HI_DEF_VIDEO.mp4)
#       DB_URL AGENT_MODEL (Qwen/Qwen3.5-4B) VLLM_GPU_UTIL (0.6) SKIP_VLLM=1 BACKEND (force) NO_PUSH=1 FORCE=1
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 15"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only || die "local branch diverged from origin; resolve manually"
[ -n "$(git log --grep='^session 14' --format=%h)" ] || die "session 14 commit not found; run build_session_14.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/preflight.sh
colab/vllm_venv.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_reid.py
tests/test_schemas.py
tests/test_slice_gpu_smoke.py
tests/test_store_agent.py
tests/test_tubes.py
vi/__init__.py
vi/agent/__init__.py
vi/agent/loop.py
vi/agent/tools.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/fake.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
vi/episode/debug.py
vi/episode/keyframes.py
vi/episode/writer.py
vi/eval/__init__.py
vi/eval/mot.py
vi/events/__init__.py
vi/events/compiler.py
vi/events/zones.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/gate/heartbeat.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/reid/__init__.py
vi/reid/base.py
vi/schemas/__init__.py
vi/schemas/common.py
vi/schemas/contact_sheet.py
vi/schemas/episode.py
vi/schemas/event.py
vi/schemas/export.py
vi/schemas/fact.py
vi/schemas/scene_card.py
vi/schemas/tick.py
vi/schemas/tube.py
vi/store/__init__.py
vi/store/db.py
vi/store/loader.py
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/linker.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p colab/sessions data/bench
cat > vi/agent/loop.py << 'EOF_VI'
"""Block 2 agent loop (R26/R27): a reasoning model that never sees video, only tool results, and
must cite ids for every claim. Every model turn is one JSON `AgentStep` under a schema, so it is
grammar-constrainable (R3) on any backend that supports structured outputs.

Backends implement `complete(messages, schema) -> str`. `FakeBackend` is a deterministic
scripted planner for tests and CPU runs; `OpenAIBackend` talks to any OpenAI-compatible server
(vLLM serve, SGLang) and asks for JSON-schema structured output.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy.engine import Engine

from . import tools as T

TOOL_SPECS = {
    "search_events": "Events by time window / zone / type / entity. args: tile_id, camera_id, t_start_ms, t_end_ms, "
                     "event_type (str or list), zone_id, entity_id, tube_id, limit",
    "search_tubes": "Tubes (per-camera tracks). args: tile_id, camera_id, class_label, t_start_ms, t_end_ms, entity_id, named, min_life_ms, limit",
    "search_entities": "Entities (people/objects across tubes). args: camera_id, class_label, t_start_ms, t_end_ms, named, limit",
    "get_script": "The scene script of an episode (cast + timeline). args: episode_id",
    "clip": "Evidence references (keyframes, time segments) for an entity or tube. args: entity_id or tube_id",
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip"]
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field("", max_length=200)


class AnswerStep(BaseModel):
    action: Literal["answer"] = "answer"
    text: str = Field(max_length=1500)
    citations: list[str] = Field(default_factory=list, description="entity ids, tube ids or event ids from tool results")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ClarifyStep(BaseModel):
    action: Literal["clarify"] = "clarify"
    question: str = Field(max_length=300)


AgentStep = Annotated[Union[ToolStep, AnswerStep, ClarifyStep], Field(discriminator="action")]
step_adapter: TypeAdapter = TypeAdapter(AgentStep)
STEP_SCHEMA = step_adapter.json_schema()

SYSTEM = """You answer questions about video by calling tools over an evidence store; you never see video.
Rules: (1) call get_script first for the episode in scope; (2) every claim in an answer must cite ids
(entity ids like cam1:E3, tube ids like cam1:4000:10, event ids like ev_...) that appeared in tool results;
(3) if nothing matches, say so and suggest how to widen the search; never invent people, times or events;
(4) if the question is ambiguous (which person, which time), ask one clarifying question;
(5) answer over entities, not tubes; a person may have several tubes; (6) times are episode-relative
mm:ss.s in scripts and absolute milliseconds in tool args. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Tools: """ + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+)\b")


class FakeBackend:
    """Scripted planner: script -> one search chosen from the question -> cited answer. Enough to
    exercise the loop, the citation check and the empty-result path without a model."""

    name = "fake"

    def __init__(self, bogus_citation: bool = False):
        self.bogus = bogus_citation

    def complete(self, messages: list[dict], schema: dict) -> str:
        user = [m for m in messages if m["role"] == "user"]
        q = user[0]["content"].lower()
        n_tool_results = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("TOOL RESULT"))
        if n_tool_results == 0:
            ep = re.search(r"episode (ep_[0-9a-f]+)", user[0]["content"], re.IGNORECASE)
            return json.dumps({"action": "tool", "tool": "get_script", "args": {"episode_id": ep.group(1) if ep else ""}, "why": "read the script"})
        if n_tool_results == 1:
            if "key" in q or "pickup" in q or "took" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "pickup"}, "why": "custody"})
            if "nobody" in q or "unicorn" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "fall"}, "why": "probe"})
            return json.dumps({"action": "tool", "tool": "search_entities", "args": {"class_label": "person"}, "why": "who"})
        last = messages[-1]["content"]
        ids = ID_RE.findall(last)
        if not ids:
            return json.dumps({"action": "answer", "text": "No matching events in this episode; widen the time window or check another tile.",
                               "citations": [], "confidence": 0.9})
        cites = ["ev_deadbeefdeadbeef"] if self.bogus else sorted(set(ids))[:4]
        return json.dumps({"action": "answer", "text": f"Found {len(set(ids))} matching record(s); see citations.",
                           "citations": cites, "confidence": 0.8})


class OpenAIBackend:
    """Any OpenAI-compatible chat server (vLLM serve, SGLang). Requests JSON-schema structured
    output; falls back to vLLM's guided_json extra field on servers that predate response_format."""

    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen3.5-4B",
                 api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 600, timeout: float = 120.0):
        import urllib.request
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self._req = urllib.request

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = self._req.Request(self.base_url + "/chat/completions", data=data, method="POST",
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with self._req.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def complete(self, messages: list[dict], schema: dict) -> str:
        base = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        try:
            out = self._post({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}})
        except Exception:
            out = self._post({**base, "guided_json": schema})
        return extract_json(out["choices"][0]["message"]["content"])


def extract_json(text: str) -> str:
    """Take the first balanced {...} object out of a model reply (code fences, prose, and
    trailing text are all tolerated). Returns the raw text if none is found, so validation
    fails loudly rather than silently."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start = t.find("{")
    if start < 0:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return text


class TransformersBackend:
    """In-process fallback when no server can be started: loads the model with transformers in the
    runtime's own torch. No structured-output grammar; the loop validates and retries instead."""

    name = "transformers"

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 600, device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        last = None
        self.model = None
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                cls = getattr(__import__("transformers", fromlist=[loader]), loader)
                self.model = cls.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                last = e
        if self.model is None:
            raise RuntimeError(f"could not load {model_id}: {last!r}")
        try:
            self.tok = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_id)

    def complete(self, messages: list[dict], schema: dict) -> str:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\nReply with the JSON object only, no prose."}
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tokenizer = getattr(self.tok, "tokenizer", self.tok)
        return extract_json(tokenizer.decode(gen, skip_special_tokens=True))


def run_tool(engine: Engine, step: ToolStep) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip}[step.tool]
    args = {k: v for k, v in step.args.items() if v is not None}
    if step.tool == "get_script":
        return fn(engine, args.get("episode_id", ""))
    return fn(engine, **args)


def _compact(result: Any, limit: int = 6000) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def ask(engine: Engine, question: str, episode_id: str, backend, max_steps: int = 8) -> dict:
    """Run the loop. Returns the final step plus the trace (every tool call and result size), the
    set of ids the model actually saw, and the citation audit (E-AGT-06)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Episode {episode_id}. Question: {question}"}]
    seen_ids: set[str] = set()
    trace: list[dict] = []
    final: dict | None = None
    for i in range(max_steps):
        raw = backend.complete(messages, STEP_SCHEMA)
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"TOOL RESULT: invalid step ({e.errors()[0]['msg']}); reply with one valid JSON object."})
            trace.append({"step": i, "invalid": raw[:200]})
            continue
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            result: Any = None
            try:
                result = run_tool(engine, step)
                text = _compact(result)
                err = None
            except Exception as e:
                text, err = f"error: {type(e).__name__}: {e}", str(e)
            seen_ids.update(ID_RE.findall(text))
            n = len(result) if isinstance(result, list) else (1 if result else 0)
            trace.append({"step": i, "tool": step.tool, "args": step.args, "results": n, "error": err})
            messages.append({"role": "user", "content": f"TOOL RESULT ({step.tool}, {n} item(s)):\n{text}"})
            continue
        if isinstance(step, ClarifyStep):
            final = {"action": "clarify", "question": step.question}
            break
        valid = [c for c in step.citations if c in seen_ids]
        invalid = [c for c in step.citations if c not in seen_ids]
        already_revised = any("revise" in t for t in trace)
        if invalid and not valid and not already_revised:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in any tool result. "
                                                        "Answer again citing only ids you were shown, or say nothing matched."})
            continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid,
                 "confidence": step.confidence, "cited": bool(valid)}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False,
                 "confidence": 0.0}
    return {"question": question, "episode_id": episode_id, "backend": getattr(backend, "name", "?"),
            "steps": len(trace), "trace": trace, "final": final}
EOF_VI
cat > vi/agent/__init__.py << 'EOF_VI'
from .loop import FakeBackend, OpenAIBackend, TransformersBackend, ask, extract_json
from .tools import clip, get_script, search_entities, search_events, search_tubes
EOF_VI
cat > bench/agent_replay.py << 'EOF_VI'
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
            res = ask(engine, q, ep, backend)
            answers.append(res)
            calls = [f"{t['tool']}({', '.join(f'{k}={v}' for k, v in t['args'].items())}) -> "
                     + (f"ERROR {t['error'][:80]}" if t.get('error') else str(t['results'])) for t in res["trace"] if "tool" in t]
            print(f"\nQ: {q}\n   tools: " + " | ".join(calls))
            f = res["final"]
            if f["action"] == "answer":
                print(f"   A ({f.get('confidence', 0):.2f}{'' if f['cited'] else ', UNCITED'}): {f['text']}\n   cites: {', '.join(f['citations']) or '—'}")
            else:
                print(f"   clarify: {f['question']}")
        row["answers"] = [{"q": r["question"], "steps": r["steps"], "cited": r["final"].get("cited"),
                           "action": r["final"]["action"], "backend": r["backend"]} for r in answers]
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "agent.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "answers"}, indent=2))


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/test_store_agent.py << 'EOF_VI'
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
    assert [t["tool"] for t in out["trace"] if "tool" in t][:2] == ["get_script", "search_events"]
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
EOF_VI
cat > colab/preflight.sh << 'EOF_VI'
#!/usr/bin/env bash
# Print what the runtime actually is before anything assumes it. Safe to run anywhere.
py() { python3 - "$@" << 'PY'
import importlib, json, shutil, subprocess, sys
r = {"python": sys.version.split()[0], "ensurepip": importlib.util.find_spec("ensurepip") is not None}
for m in ("torch", "torchaudio", "torchvision", "transformers", "vllm", "rfdetr", "av", "sqlalchemy", "psycopg"):
    try:
        mod = importlib.import_module(m)
        r[m] = getattr(mod, "__version__", "?")
        if m == "torch":
            r["torch_cuda"] = getattr(mod.version, "cuda", None)
            r["cuda_available"] = bool(mod.cuda.is_available())
            if mod.cuda.is_available():
                free, total = mod.cuda.mem_get_info()
                r["gpu"] = f"{mod.cuda.get_device_name(0)} free {free/1e9:.1f}/{total/1e9:.1f} GB"
    except Exception as e:
        r[m] = f"absent ({type(e).__name__})"
r["uv"] = shutil.which("uv") is not None
r["disk_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1)
print(json.dumps(r))
PY
}
py
EOF_VI
cat > colab/vllm_venv.sh << 'EOF_VI'
#!/usr/bin/env bash
# vLLM in its own environment so its torch never touches the runtime's (Colab ships a CUDA-13 torch;
# vLLM wheels bring a CUDA-12.8 torch and torchaudio refuses to import next to it).
# Colab's python has no ensurepip, so the venv is built by, in order: uv -> venv --without-pip +
# pip bootstrap -> virtualenv. Each failure is printed and the next method is tried.
#   bash colab/vllm_venv.sh start [model] [port]   # install if needed, serve in background, wait
#   bash colab/vllm_venv.sh stop
#   bash colab/vllm_venv.sh venv                    # only build the venv (smoke test)
set -uo pipefail
VENV="${VLLM_VENV:-/content/vllm-venv}"
MODEL="${2:-${AGENT_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${3:-8000}"
LOG="/tmp/vllm.log"
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; echo "stopped"; exit 0; fi
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }

make_venv() {
  [ -x "$VENV/bin/python" ] && return 0
  echo "-- venv via uv"
  if (command -v uv >/dev/null 2>&1 || pipi uv) && uv venv "$VENV" --python "$(command -v python3)" >/tmp/venv.log 2>&1 \
     && uv pip install --python "$VENV/bin/python" -q pip >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via python -m venv --without-pip + get-pip"
  if python3 -m venv --without-pip "$VENV" >/tmp/venv.log 2>&1 \
     && curl -sSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
     && "$VENV/bin/python" /tmp/get-pip.py -q >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  echo "-- venv via virtualenv"
  if pipi virtualenv && python3 -m virtualenv -q "$VENV" >/tmp/venv.log 2>&1; then echo "   ok"; return 0; fi
  echo "   failed: $(tail -1 /tmp/venv.log)"; rm -rf "$VENV"
  return 1
}

install_vllm() {
  "$VENV/bin/python" -c "import vllm" 2>/dev/null && return 0
  echo "-- installing vllm into the venv (isolated torch; 3-6 min)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv)"; return 0; fi
  if "$VENV/bin/python" -m pip install -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (pip)"; return 0; fi
  echo "   failed: $(tail -2 /tmp/vllm_install.log | tr '\n' ' ')"; return 1
}

make_venv || { echo "could not create a virtualenv by any method"; exit 1; }
if [ "$cmd" = venv ]; then "$VENV/bin/python" -c "import sys; print('   venv python', sys.version.split()[0])"; exit 0; fi
install_vllm || exit 1
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
  --max-model-len 16384 --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.6}" --dtype bfloat16 --max-num-seqs 4 > "$LOG" 2>&1 &
echo "-- waiting for http://127.0.0.1:$PORT (weights download on first run)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   vLLM up: $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; exit 0
  fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "   server exited; log tail:"; tail -12 "$LOG"; exit 1; fi
  sleep 5
done
echo "   server did not come up in 10 min; log tail:"; tail -12 "$LOG"; exit 1
EOF_VI
chmod +x colab/preflight.sh colab/vllm_venv.sh
cp "$0" colab/sessions/session_15_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. preflight (what this runtime actually is)"
bash colab/preflight.sh | tee data/bench/preflight.json

step "4. install + harness"
pipi -e ".[dev,ingest]"
make check

step "5. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then
    echo "installing PostgreSQL (~1-2 min)"
    (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed: $(tail -2 /tmp/apt.log | tr '\n' ' ')"
  fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" && sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"
    DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
    echo "PostgreSQL: $(sudo -u postgres psql -tAc 'select version()' | cut -d, -f1)"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "6. slice (frame + SigLIP) -> episode -> store"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|entities|relinks)\"" /tmp/slice_final.log | sed 's/^/  /'
else
  warn "no GPU or no clip; using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "7. reasoning backend: vLLM venv -> transformers in-process -> fake"
BACKEND="${BACKEND:-}"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ -z "$BACKEND" ]; then
  if [ "$HAS_GPU" = 1 ] && [ "${SKIP_VLLM:-0}" != "1" ]; then
    if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai; else warn "vLLM path failed -> trying transformers in-process"; fi
  fi
  if [ -z "$BACKEND" ] && [ "$HAS_GPU" = 1 ]; then
    python -c "import transformers" 2>/dev/null || pipi transformers
    BACKEND=transformers          # agent_replay falls back to fake by itself if the model cannot load
  fi
  [ -z "$BACKEND" ] && { BACKEND=fake; warn "no GPU: fake backend"; }
fi
echo "backend: $BACKEND ($AGENT_MODEL)"

step "8. three questions"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 0 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was in the exit_right zone, when did they arrive, and is there a keyframe for them?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]"
[ "$BACKEND" = openai ] && bash colab/vllm_venv.sh stop >/dev/null 2>&1 || true

step "9. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: preflight, venv builder chain (uv/without-pip/virtualenv), in-process transformers backend, backend fallback chain"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "10. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 3, 7 and 8 back (preflight line, backend chosen, the three Q/A blocks)."
