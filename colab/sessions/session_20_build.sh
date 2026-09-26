#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 20 build:
#   * vLLM: torch build = the driver's exact CUDA (cu130, not cu132), gpu-memory-utilization 0.85,
#     max-model-len 8192; on failure the log is printed from the first ERROR line to the end
#   * agent: event-claim check (an answer naming an event type the episode has none of is sent back
#     once; negated mentions are fine); citations flattened from objects/prose; answers <= 80 words;
#     transformers backend warms up before the first question
#   * scenario: the avoids rule uses stand-alone numbers too (00:14.0 is not "14")
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_20.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 20"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] || warn "GH_TOKEN is not set: will commit locally but cannot push (use the one-cell form)"

step "0. runtime checks"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ "$HAS_GPU" = 1 ]; then echo "GPU: $(nvidia-smi -L | head -1)"; echo "driver: $(nvidia-smi | grep -oE 'Driver Version: [0-9.]+|CUDA Version: [0-9.]+' | tr '\n' ' ')"
else warn "NO GPU in this runtime -> Runtime > Change runtime type > L4, then rerun"; fi
[ -f "$SOURCE" ] && echo "clip: $SOURCE ($(du -h "$SOURCE" | cut -f1))" || warn "NO CLIP at $SOURCE -> upload HI_DEF_VIDEO.mp4 to /content"

step "1. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin || true
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"
if [ -n "$(git status --porcelain)" ]; then
  warn "uncommitted changes from an interrupted run; discarding them (the script rewrites its files)"
  git checkout -- . && git clean -fdq
fi
git pull -q --ff-only 2>/dev/null || warn "pull skipped (offline or diverged); continuing on local HEAD"
[ -n "$(git log --grep='^session 19' --format=%h)" ] || die "session 19 commit not found; run build_session_19.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/cast_sheet.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/scenario_eval.py
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
tests/scenarios/warehouse.yaml
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
vi/tubes/quality.py
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
cat > colab/vllm_venv.sh << 'EOF_VI'
#!/usr/bin/env bash
# vLLM in its own environment so its torch never touches the runtime's (Colab ships a CUDA-13 torch;
# vLLM wheels bring a CUDA-12.8 torch and torchaudio refuses to import next to it).
# Colab's python has no ensurepip, so the venv is built by, in order: uv -> venv --without-pip +
# pip bootstrap -> virtualenv. Each failure is printed and the next method is tried.
#   bash colab/vllm_venv.sh start [model] [port]   # install if needed, serve in background, wait
#   bash colab/vllm_venv.sh stop
#   bash colab/vllm_venv.sh status                  # up | down
#   bash colab/vllm_venv.sh venv                    # only build the venv (smoke test)
set -uo pipefail
VENV="${VLLM_VENV:-/content/vllm-venv}"
MODEL="${2:-${AGENT_MODEL:-Qwen/Qwen3.5-4B}}"
PORT="${3:-8000}"
LOG="/tmp/vllm.log"
cmd="${1:-start}"
if [ "$cmd" = stop ]; then pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; echo "stopped"; exit 0; fi
if [ "$cmd" = status ]; then curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "up" || { echo "down"; exit 1; }; exit 0; fi
if [ "$cmd" = start ] && curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  served="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
  if [ "$served" = "$MODEL" ]; then echo "   vLLM already up: $served (reusing; run 'stop' to restart)"; exit 0; fi
  echo "   vLLM up with $served, restarting for $MODEL"; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; sleep 3
fi
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

driver_cuda() {  # e.g. 12.8 from nvidia-smi; the torch build must not be newer than this
  nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1
}

install_vllm() {
  if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" -c "import vllm" >/tmp/vllm_import.log 2>&1; then return 0; fi
    echo "-- venv exists but 'import vllm' fails ($(grep -oE "ImportError: [^\n]*|ModuleNotFoundError: [^\n]*" /tmp/vllm_import.log | head -1 | cut -c1-100)): rebuilding"
    rm -rf "$VENV"; make_venv || return 1
  fi
  drv="$(driver_cuda)"
  # vLLM's PyPI wheel links libcudart of a fixed CUDA major (13 for 0.30); the torch build must match
  # the driver's CUDA, not the runtime's torch (session 18 forced cu128 and broke vllm._C).
  drv_tag="$(echo "$drv" | tr -d .)"                       # 13.0 -> cu130: the exact build for this driver
  backend="${VLLM_TORCH_BACKEND:-${drv_tag:+cu$drv_tag}}"; backend="${backend:-auto}"
  echo "-- installing vllm into the venv (driver CUDA ${drv:-?}, torch backend $backend; 3-6 min)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q --torch-backend="$backend" vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv, torch-backend=$backend)"; return 0; fi
  echo "   uv with --torch-backend=$backend failed: $(tail -1 /tmp/vllm_install.log | cut -c1-120)"
  if command -v uv >/dev/null 2>&1 && uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (uv, torch-backend=auto)"; return 0; fi
  if "$VENV/bin/python" -m pip install -q vllm >/tmp/vllm_install.log 2>&1; then echo "   ok (pip)"; return 0; fi
  echo "   failed: $(tail -2 /tmp/vllm_install.log | tr '\n' ' ')"; return 1
}

root_cause() {  # everything from the first ERROR/Traceback to the end of the log, INFO lines dropped
  local n; n="$(grep -nE "ERROR|Traceback|Error:" "$LOG" | head -1 | cut -d: -f1)"
  if [ -n "$n" ]; then tail -n "+$n" "$LOG" | grep -v "INFO\|TracerWarning" | tail -40 | cut -c1-220
  else tail -25 "$LOG" | cut -c1-220; fi
}

make_venv || { echo "could not create a virtualenv by any method"; exit 1; }
if [ "$cmd" = venv ]; then "$VENV/bin/python" -c "import sys; print('   venv python', sys.version.split()[0])"; exit 0; fi
install_vllm || exit 1
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" || exit 1
echo "   driver CUDA $(driver_cuda)"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
# clean process environment: no PYTHONPATH, no user site, and only the driver's library dir on LD_LIBRARY_PATH
# (Colab's LD_LIBRARY_PATH points at the main environment's CUDA libraries, which can shadow the venv's)
nohup env -u PYTHONPATH PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/usr/lib64-nvidia}" \
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
  --max-model-len "${VLLM_MAX_LEN:-8192}" --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" --dtype bfloat16 --max-num-seqs 4 \
  --enable-prefix-caching > "$LOG" 2>&1 &
echo "-- waiting for http://127.0.0.1:$PORT (weights download on first run)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   vLLM up: $(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; exit 0
  fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "   server exited; root cause:"; root_cause; exit 1; fi
  sleep 5
done
echo "   server did not come up in 10 min; root cause:"; root_cause; exit 1
EOF_VI
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
import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy.engine import Engine

from vi.schemas import EventType

from . import tools as T

EVENT_TYPES = [e.value for e in EventType]

TOOL_SPECS = {
    "search_events": "Events by time window / zone / type / entity. args: tile_id, camera_id, t_start_ms, t_end_ms, "
                     "event_type (one of EVENT_TYPES, or a list), zone_id, entity_id, tube_id, limit",
    "search_tubes": "Tubes (per-camera tracks). args: tile_id, camera_id, class_label, t_start_ms, t_end_ms, entity_id, named, min_life_ms, limit",
    "search_entities": "Entities (people/objects across tubes). args: camera_id, class_label, t_start_ms, t_end_ms, named, limit",
    "get_script": "The scene script of an episode (cast + timeline). args: episode_id",
    "clip": "Evidence references (keyframes, time segments) for an entity or tube. args: entity_id or tube_id",
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip"]
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field("", max_length=400)


ANSWER_MAX_CHARS = 4000


class AnswerStep(BaseModel):
    action: Literal["answer"] = "answer"
    text: str = Field(max_length=ANSWER_MAX_CHARS)
    citations: list[str] = Field(default_factory=list, description="entity ids, tube ids or event ids from tool results")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ClarifyStep(BaseModel):
    action: Literal["clarify"] = "clarify"
    question: str = Field(max_length=300)


AgentStep = Annotated[Union[ToolStep, AnswerStep, ClarifyStep], Field(discriminator="action")]
step_adapter: TypeAdapter = TypeAdapter(AgentStep)
STEP_SCHEMA = step_adapter.json_schema()

SYSTEM = """You answer questions about video by calling tools over an evidence store; you never see video.
Rules: (1) the episode's scene script is given to you; call tools only when it does not answer the question;
(2) every claim in an answer must cite ids
(entity ids like cam1:E3, tube ids like cam1:4000:10, event ids like ev_...) that appeared in tool results;
(3) if nothing matches, say so and suggest how to widen the search; never invent people, times or events;
(4) if the question is ambiguous (which person, which time), ask one clarifying question;
(5) answer over entities, not tubes; a person may have several tubes; (6) times are episode-relative
mm:ss.s in scripts and absolute milliseconds in tool args. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Count people from CONFIRMED entities; BRIEF SIGHTINGS are not people.
Keep answers under 80 words; mention only events that appear in the script or tool results.
EVENT_TYPES: """ + ", ".join(EVENT_TYPES) + "\nTools: " + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+)\b")


class FakeBackend:
    """Scripted planner: script -> one search chosen from the question -> cited answer. Enough to
    exercise the loop, the citation check and the empty-result path without a model."""

    name = "fake"

    def __init__(self, bogus_citation: bool = False):
        self.bogus = bogus_citation

    def complete(self, messages: list[dict], schema: dict) -> str:
        user = [m for m in messages if m["role"] == "user"]
        q = user[0]["content"].split("Question:")[-1].lower()     # the question, not the inlined script
        n_tool_results = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("TOOL RESULT"))
        has_script = "SCENE SCRIPT:" in user[0]["content"]
        if n_tool_results == 0 and not has_script:
            ep = re.search(r"episode (ep_[0-9a-f]+)", user[0]["content"], re.IGNORECASE)
            return json.dumps({"action": "tool", "tool": "get_script", "args": {"episode_id": ep.group(1) if ep else ""}, "why": "read the script"})
        if n_tool_results == (0 if has_script else 1):
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
        base = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}      # Qwen3.x: no reasoning preamble before the JSON
        try:
            out = self._post({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}})
        except Exception:
            try:
                out = self._post({**base, "guided_json": schema})
            except Exception:
                base.pop("chat_template_kwargs", None)                   # servers that reject the field
                out = self._post({**base, "guided_json": schema})
        return extract_json(out["choices"][0]["message"]["content"])


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> str:
    """Take the first balanced {...} object out of a model reply (think blocks, code fences,
    prose, and trailing text are all tolerated). Returns the raw text if none is found, so
    validation fails loudly rather than silently."""
    t = THINK_RE.sub("", text).strip()
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

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 300, device: str | None = None, warmup: bool = True):
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
        if warmup:   # CUDA context, kernels and cache allocation happen here, not inside the first question
            try:
                self.complete([{"role": "system", "content": "Reply with {}"}, {"role": "user", "content": "{}"}], {})
            except Exception:
                pass

    def complete(self, messages: list[dict], schema: dict) -> str:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\nReply with the JSON object only, no prose."}
        try:   # Qwen3.x templates: thinking is on by default and eats the whole token budget
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tokenizer = getattr(self.tok, "tokenizer", self.tok)
        return extract_json(tokenizer.decode(gen, skip_special_tokens=True))


def _flatten_ids(x: Any) -> list[str]:
    """citations may arrive as strings, ints, dicts ({"entity_id": ...}) or nested lists"""
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return ID_RE.findall(str(x)) or ([str(x)] if isinstance(x, str) else [])
    if isinstance(x, dict):
        return [i for v in x.values() for i in _flatten_ids(v)]
    if isinstance(x, list):
        return [i for v in x for i in _flatten_ids(v)]
    return []


EVENT_WORDS = {"pickup": ["pickup", "pick-up", "picked up", "picking up", "picks up"], "drop": ["drop event", "dropped"],
               "fall": ["fall event", "fell", "falling"], "left_behind": ["left behind"], "loiter": ["loiter"],
               "crowd": ["crowd event"], "run": ["running event"], "handoff": ["handoff", "hand-off"],
               "impossible_transition": ["impossible transition"], "relink": ["relink"]}


def contradicted_event_claims(engine: Engine, episode_id: str, text: str) -> list[str]:
    """E-AGT-06 for prose: an answer that talks about an event type the episode does not contain
    is asserting evidence that does not exist. Returns the offending event types."""
    # sentence-level, ignoring negated mentions ("nobody fell", "no pickup events") which assert absence
    neg = re.compile(r"\b(no|not|nobody|none|never|didn't|did not|wasn't|weren't|isn't|aren't|without)\b")
    named: set[str] = set()
    for sent in re.split(r"(?<=[.!?;])\s+", text.lower()):
        if neg.search(sent):
            continue
        for et, words in EVENT_WORDS.items():
            if any(w in sent for w in words):
                named.add(et)
    if not named:
        return []
    have = {e["type"] for e in T.search_events(engine, limit=10000) if e["episode_id"] == episode_id}
    return sorted(et for et in named if et not in have)


def _salvage(raw: str) -> AnswerStep | ClarifyStep | None:
    """Accept an answer that only failed on length or a missing optional field; never a tool
    call, which must be exact. Retrying seven times to trim a paragraph is not a behaviour."""
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("action") == "answer" and isinstance(obj.get("text"), str):
        cites = _flatten_ids(obj.get("citations")) + ID_RE.findall(obj["text"])   # objects, nested lists, ids in prose
        seen: list[str] = []
        for c in cites:
            if c not in seen:
                seen.append(c)
        return AnswerStep(text=obj["text"][:ANSWER_MAX_CHARS], citations=seen[:50],
                          confidence=float(obj.get("confidence", 0.5)) if isinstance(obj.get("confidence"), (int, float)) else 0.5)
    if obj.get("action") == "clarify" and isinstance(obj.get("question"), str):
        return ClarifyStep(question=obj["question"][:300])
    return None


def run_tool(engine: Engine, step: ToolStep) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip}[step.tool]
    import inspect
    allowed = set(inspect.signature(fn).parameters) - {"engine", "episode_id"} if step.tool != "get_script" else {"episode_id"}
    dropped = [k for k in step.args if k not in allowed]
    args = {k: v for k, v in step.args.items() if v is not None and k in allowed}
    if step.tool == "search_events" and args.get("event_type"):
        types = args["event_type"] if isinstance(args["event_type"], list) else [args["event_type"]]
        bad = [t for t in types if t not in EVENT_TYPES]
        if bad:
            raise ValueError(f"unknown event_type {bad}; valid: {', '.join(EVENT_TYPES)}")
    if step.tool == "get_script":
        return fn(engine, args.get("episode_id", ""))
    result = fn(engine, **args)
    if dropped and isinstance(result, list):
        result = [{"note": f"ignored unknown args {dropped}"}] + result if result else [{"note": f"ignored unknown args {dropped}; no matches"}]
    return result


def _compact(result: Any, limit: int = 6000) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def ask(engine: Engine, question: str, episode_id: str, backend, max_steps: int = 6,
        inline_script: bool = True) -> dict:
    """Run the loop. With inline_script the scene script is placed in the first user message so
    the model's first turn is already a search or an answer (one round trip saved) and the
    server's prefix cache holds system+script across questions on the same episode. Returns the
    final step, the trace, the citation audit (E-AGT-06) and latency."""
    t_start = time.perf_counter()
    seen_ids: set[str] = set()
    first = f"Episode {episode_id}."
    if inline_script:
        script = T.get_script(engine, episode_id)
        seen_ids.update(ID_RE.findall(script))
        first += f"\n\nSCENE SCRIPT:\n{script}"
    first += f"\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    trace: list[dict] = []
    final: dict | None = None
    model_ms, tool_ms = 0.0, 0.0
    for i in range(max_steps):
        t0 = time.perf_counter()
        raw = backend.complete(messages, STEP_SCHEMA)
        model_ms += (time.perf_counter() - t0) * 1000
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            step = _salvage(raw)                    # an over-long or slightly malformed answer is still an answer
            if step is None:
                reason = e.errors()[0]["msg"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"TOOL RESULT: invalid step ({reason}); reply with one valid JSON object."})
                trace.append({"step": i, "invalid": raw[:200], "reason": reason})
                continue
            trace.append({"step": i, "salvaged": e.errors()[0]["msg"]})
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            result: Any = None
            t0 = time.perf_counter()
            try:
                result = run_tool(engine, step)
                text = _compact(result)
                err = None
            except Exception as e:
                text, err = f"error: {type(e).__name__}: {e}", str(e)
            tool_ms += (time.perf_counter() - t0) * 1000
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
        bad_claims = contradicted_event_claims(engine, episode_id, step.text)
        if bad_claims and not already_revised:
            trace.append({"step": i, "revise": "event claim without evidence", "claims": bad_claims})
            messages.append({"role": "user", "content": f"TOOL RESULT: your answer mentions {bad_claims} but this episode contains no "
                                                        f"events of that type. Remove or correct that claim; cite only events that exist."})
            continue
        if invalid and not valid and not already_revised:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in any tool result. "
                                                        "Answer again citing only ids you were shown, or say nothing matched."})
            continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid,
                 "confidence": step.confidence, "cited": bool(valid), "unsupported_event_claims": bad_claims}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False,
                 "confidence": 0.0}
    total_ms = (time.perf_counter() - t_start) * 1000
    return {"question": question, "episode_id": episode_id, "backend": getattr(backend, "name", "?"),
            "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round(total_ms), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1,
                        "first_prompt_chars": len(first)}}
EOF_VI
cat > bench/scenario_eval.py << 'EOF_VI'
"""Score the agent against a ground-truth scenario file. Every question gets pass/fail on the
mention rules, the citation rules and the latency budget; the row goes to data/bench/scenario.jsonl.

  python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend openai
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, search_events
from vi.agent.loop import ID_RE
from vi.store import connect


import re


def _one(r, text: str) -> bool:
    r = str(r).lower()
    if r.isdigit():                       # a number must stand alone: not part of 00:14.0, E14 or 140
        return re.search(rf"(?<![\d:.\w]){re.escape(r)}(?![\d:.\w])", text) is not None
    return r in text


def _mentioned(rule, text: str) -> bool:
    """a string must appear; a list means any of its strings must appear"""
    if isinstance(rule, list):
        return any(_one(r, text) for r in rule)
    return _one(rule, text)


def score(res: dict, spec: dict, budget_ms: int) -> dict:
    f = res["final"]
    text = ID_RE.sub(" ", f.get("text") or f.get("question") or "").lower()    # ids are not numbers
    checks = {
        "answered": f["action"] == "answer",
        "mentions": all(_mentioned(m, text) for m in spec.get("must_mention", []) or []),
        "avoids": not any(_one(m, text) for m in spec.get("must_not_mention", []) or []),
        "cites": (f.get("cited", False) or spec.get("expect_uncited_ok", False))
                 and all(any(c.startswith(p) for c in f.get("citations", [])) for p in spec.get("must_cite_prefix", []) or []),
        "in_budget": res["latency"]["total_ms"] <= budget_ms,
    }
    return {"q": spec["q"], "pass": all(checks.values()), "checks": checks, "latency_ms": res["latency"]["total_ms"],
            "answer": f.get("text", f.get("question", ""))[:300], "citations": f.get("citations", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--db", required=True)
    ap.add_argument("--episode", default=None, help="default: the latest episode in the store")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.scenario).read_text())
    engine = connect(a.db)
    ep = a.episode or sorted({e["episode_id"] for e in search_events(engine, limit=10000)})[-1]
    backend = {"fake": lambda: FakeBackend(), "openai": lambda: OpenAIBackend(base_url=a.base_url, model=a.model),
               "transformers": lambda: TransformersBackend(model_id=a.model)}[a.backend]()
    budget = int(spec.get("latency_budget_ms", 10000))
    results = [score(ask(engine, q["q"], ep, backend), q, budget) for q in spec["questions"]]
    unscored = spec.get("ground_truth", {}).get("people_total") is None
    row = {"ring": "scenario", "scenario": Path(a.scenario).name, "episode_id": ep, "backend": a.backend, "model": a.model,
           "passed": sum(r["pass"] for r in results), "total": len(results), "ground_truth_filled": not unscored,
           "latency_p50_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
           "latency_max_ms": max(r["latency_ms"] for r in results), "results": results,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "scenario.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    for r in results:
        flags = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["checks"].items())
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['latency_ms'] / 1000:5.1f}s  {r['q'][:70]}\n         {flags}\n         {r['answer'][:160]}")
    print(f"  {row['passed']}/{row['total']} passed, p50 {row['latency_p50_ms'] / 1000:.1f}s, max {row['latency_max_ms'] / 1000:.1f}s"
          + ("  (ground truth not filled in yet: mention rules are placeholders)" if unscored else ""))


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


def test_think_blocks_are_stripped_and_unknown_args_dropped(episode_path):
    from vi.agent import extract_json
    from vi.agent.loop import ToolStep, run_tool
    assert extract_json('<think>\nlet me reason...\n{"not": "this"}\n</think>\n{"action":"clarify","question":"which?"}') \
        == '{"action":"clarify","question":"which?"}'
    engine = connect()
    load_episode_file(engine, episode_path)
    out = run_tool(engine, ToolStep(tool="search_events", args={"episode_id": "ep_x", "event_type": "pickup", "bogus": 1}))
    assert out[0]["note"].startswith("ignored unknown args") and "episode_id" in out[0]["note"] and len(out) == 2


def test_overlong_answer_is_salvaged_not_retried(episode_path):
    from vi.agent import ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]

    class LongWinded:
        name = "long"
        def complete(self, messages, schema):
            import json
            return json.dumps({"action": "answer", "text": "x" * 6000, "citations": ["cam1:2000:1"], "confidence": 0.7})
    out = ask(engine, "anything", ep, LongWinded())
    assert out["final"]["action"] == "answer" and len(out["final"]["text"]) == 4000 and out["final"]["cited"]
    assert out["steps"] == 1 and "salvaged" in out["trace"][0]


def test_scenario_numbers_must_stand_alone():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("scenario_eval", "bench/scenario_eval.py")
    m = importlib.util.module_from_spec(spec); sys.modules["scenario_eval"] = m; spec.loader.exec_module(m)
    assert m._mentioned("14", "left at 00:14.0 and e14 and 140 people") is False
    assert m._mentioned("14", "there were 14 people") is True
    assert m._mentioned(["8", "eight"], "eight workers") is True
    assert m._one("14", "cam1:e10 left at 00:14.0") is False       # the avoids rule uses the same test


@pytest.mark.edge("E-AGT-06")
def test_unsupported_event_claims_are_sent_back_and_reported(episode_path):
    from vi.agent import ask
    engine = connect()
    load_episode_file(engine, episode_path)
    ep = search_events(engine)[0]["episode_id"]

    class Fabricator:
        name = "fab"
        def __init__(self): self.n = 0
        def complete(self, messages, schema):
            import json
            self.n += 1
            if self.n == 1:
                return json.dumps({"action": "answer", "text": "Nobody fell. But cam1:2000:1 was seen falling at 00:09.",
                                   "citations": [{"entity_id": "cam1:2000:1"}], "confidence": 0.9})
            assert "fall" in messages[-1]["content"]
            return json.dumps({"action": "answer", "text": "Nobody fell. cam1:2000:1 picked up the bike keys.",
                               "citations": ["cam1:2000:1"], "confidence": 0.9})
    out = ask(engine, "Did anyone fall?", ep, Fabricator())
    assert any(t.get("revise") == "event claim without evidence" and t["claims"] == ["fall"] for t in out["trace"])
    assert out["final"]["unsupported_event_claims"] == [] and out["final"]["cited"]      # pickup exists in this episode


def test_salvage_flattens_citation_objects_and_ids_in_prose():
    from vi.agent.loop import _salvage
    import json
    step = _salvage(json.dumps({"action": "answer", "text": "cam1:E10 arrived at 00:04.0 [ev_0123456789abcdef]",
                                "citations": [{"entity_id": "cam1:E10"}, ["cam1:4000:10"], 7]}))
    assert step is not None and step.citations == ["cam1:E10", "cam1:4000:10", "ev_0123456789abcdef"]
EOF_VI
chmod +x colab/vllm_venv.sh colab/preflight.sh 2>/dev/null || true
cp "$0" colab/sessions/session_20_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"
  fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" 2>/dev/null; sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"
    DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "5. slice (frame + SigLIP, real zones) -> episode -> store -> cast sheet"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes data/keyframes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|person_tubes_low_quality|entities|relinks|max_concurrent_persons)\"" /tmp/slice_final.log | sed 's/^/  /'
  python bench/cast_sheet.py data/episodes/*.jsonl --keyframes data/keyframes --out data/bench/cast_sheet.jpg
  echo "  -> upload /content/Tracer/data/bench/cast_sheet.jpg to the chat (Files panel > right-click > Download)"
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "6. reasoning backend: vLLM (cu<driver>, 0.85 GPU, 8k ctx, clean env) -> transformers -> fake"
BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if [ "$HAS_GPU" = 1 ] && [ "${SKIP_VLLM:-0}" != "1" ]; then
    if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai; else warn "vLLM path failed -> transformers in-process"; fi
  fi
  if [ -z "$BACKEND" ] && [ "$HAS_GPU" = 1 ]; then python -c "import transformers" 2>/dev/null || pipi transformers; BACKEND=transformers; fi
  [ -z "$BACKEND" ] && { BACKEND=fake; warn "no GPU: fake backend"; }
fi
echo "backend: $BACKEND ($AGENT_MODEL)"

step "7. the three questions with latency"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 3 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -v "^\s*\"" | grep -v "^[{}]" | grep -v "HF Hub"

step "8. scenario acceptance (8 people, 6 whole time, 0 pickups, 10 s budget)"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -v "HF Hub\|Loading weights" | tail -12

step "9. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: exact-driver vLLM build + memory/context settings + full root cause, event-claim check, citation flattening, avoids fix, warmup"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "10. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 6, 7, 8 back and upload data/bench/cast_sheet.jpg."
