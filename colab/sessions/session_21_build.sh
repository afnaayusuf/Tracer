#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 21 build: make the Colab path fast enough without vLLM.
#   * vLLM: ONE more attempt, with the venv rebuilt for the exact driver CUDA and the engine's
#     own error printed first. If it fails, the script records it and stops trying (VLLM_TRIED).
#   * transformers path: flash-linear-attention fused kernels (Qwen3.5 GDN layers), sdpa, tokens/s
#     reported, answers <= 60 words citing entity ids
#   * acceptance run with Qwen3.5-4B, then with Qwen3.5-2B; the two summaries decide the model
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_21.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
ALT_MODEL="${ALT_MODEL:-Qwen/Qwen3.5-2B}"
SESSION="session 21"
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
[ -n "$(git log --grep='^session 20' --format=%h)" ] || die "session 20 commit not found; run build_session_20.sh first"
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
  drv="$(driver_cuda)"
  if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" -c "import vllm" >/tmp/vllm_import.log 2>&1; then
      have="$("$VENV/bin/python" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)"
      if [ -n "$have" ] && [ -n "$drv" ] && [ "$have" != "$drv" ] && [ "${VLLM_KEEP_VENV:-0}" != "1" ]; then
        echo "-- venv torch is cu$have but the driver is CUDA $drv: rebuilding with the exact driver build"
        rm -rf "$VENV"; make_venv || return 1
      else
        return 0
      fi
    else
      echo "-- venv exists but 'import vllm' fails ($(grep -oE "ImportError: [^\n]*|ModuleNotFoundError: [^\n]*" /tmp/vllm_import.log | head -1 | cut -c1-100)): rebuilding"
      rm -rf "$VENV"; make_venv || return 1
    fi
  fi
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

root_cause() {  # the ENGINE's error (EngineCore lines) comes first in the log; the API server's stack only repeats it
  local n; n="$(grep -nE "EngineCore.*(ERROR|Error|Traceback)|ERROR|Traceback|Error:" "$LOG" | head -1 | cut -d: -f1)"
  if [ -n "$n" ]; then tail -n "+$n" "$LOG" | grep -v "INFO\|TracerWarning\|APIServer" | head -45 | cut -c1-220
  else tail -25 "$LOG" | cut -c1-220; fi
  echo "   (full log: $LOG)"
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
Keep answers under 60 words; cite entity ids (cam1:E7), not tube ids, unless asked about tubes; mention only events
that appear in the script or tool results.
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
        try:
            import fla  # noqa: F401  (flash-linear-attention: fused kernels for Qwen3.5's GDN layers)
            self.fused = True
        except Exception:
            self.fused = False
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                cls = getattr(__import__("transformers", fromlist=[loader]), loader)
                try:
                    self.model = cls.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
                except Exception:
                    self.model = cls.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                last = e
        self.last_gen_tokens = 0
        self.last_tok_s = 0.0
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
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        dt = max(1e-6, time.perf_counter() - t0)
        self.last_gen_tokens = int(getattr(gen, "shape", [len(gen)])[0])
        self.last_tok_s = self.last_gen_tokens / dt
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
                        "first_prompt_chars": len(first),
                        "gen_tokens": getattr(backend, "last_gen_tokens", None), "tok_s": round(getattr(backend, "last_tok_s", 0.0), 1) or None,
                        "fused_kernels": getattr(backend, "fused", None)}}
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
                  f"{lat['turns']} turn(s), first prompt {lat['first_prompt_chars']} chars"
                  + (f", {lat['gen_tokens']} tokens @ {lat['tok_s']} tok/s, fused={lat['fused_kernels']}" if lat.get('tok_s') else "")
                  + ")\n   tools: " + (" | ".join(calls) or "none"))
            bad = [t for t in res["trace"] if "invalid" in t or "salvaged" in t]
            if bad:
                print("   " + " | ".join(f"invalid: {t['reason'][:90]}" if "invalid" in t else f"salvaged: {t['salvaged'][:90]}" for t in bad[:3]))
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
EOF_VI
chmod +x colab/vllm_venv.sh colab/preflight.sh 2>/dev/null || true
cp "$0" colab/sessions/session_21_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness (+ fused linear-attention kernels for the transformers path)"
pipi -e ".[dev,ingest]"
if [ "$HAS_GPU" = 1 ]; then
  python -c "import fla" 2>/dev/null || { echo "installing flash-linear-attention"; pipi flash-linear-attention >/tmp/fla.log 2>&1 && echo "   ok" || warn "flash-linear-attention install failed: $(tail -1 /tmp/fla.log | cut -c1-120)"; }
  python -c "import fla; print('   fla', getattr(fla, '__version__', '?'))" 2>/dev/null || warn "fused kernels unavailable; the unfused fallback is slower"
fi
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

step "5. slice (frame + SigLIP, real zones) -> episode -> store"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes data/keyframes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --max-frames 400 \
    > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "\"(person_tubes|person_tubes_low_quality|entities|max_concurrent_persons)\"" /tmp/slice_final.log | sed 's/^/  /'
  python bench/cast_sheet.py data/episodes/*.jsonl --keyframes data/keyframes --out data/bench/cast_sheet.jpg >/dev/null && echo "  cast sheet: data/bench/cast_sheet.jpg (please upload it)"
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "6. reasoning backend: one diagnosed vLLM attempt, else transformers"
BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if [ "$HAS_GPU" = 1 ] && [ "${SKIP_VLLM:-0}" != "1" ] && [ ! -f data/bench/VLLM_TRIED ]; then
    if bash colab/vllm_venv.sh start "$AGENT_MODEL" 8000; then BACKEND=openai
    else
      warn "vLLM failed again on this runtime; recording it and not retrying (delete data/bench/VLLM_TRIED to retry)"
      echo "$(date -u +%FT%TZ) $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+')" > data/bench/VLLM_TRIED
    fi
  elif [ -f data/bench/VLLM_TRIED ]; then echo "vLLM previously failed here ($(cat data/bench/VLLM_TRIED)); using transformers"; fi
  if [ -z "$BACKEND" ] && [ "$HAS_GPU" = 1 ]; then BACKEND=transformers; fi
  [ -z "$BACKEND" ] && { BACKEND=fake; warn "no GPU: fake backend"; }
fi
echo "backend: $BACKEND"

step "7. the three questions, $AGENT_MODEL (with tokens/s)"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 0 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -E "^Q:|latency:|tools:|salvaged|invalid|A \(|clarify|cites:"

step "8. scenario acceptance: $AGENT_MODEL vs $ALT_MODEL (10 s budget)"
echo "-- $AGENT_MODEL"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -E "^\s+\[(PASS|FAIL)\]|passed," 
if [ "$BACKEND" = transformers ]; then
  echo "-- $ALT_MODEL"
  python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend transformers --model "$ALT_MODEL" 2>&1 | grep -E "^\s+\[(PASS|FAIL)\]|passed,|\[agent\]"
fi

step "9. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: vLLM rebuild-on-mismatch + engine-first root cause (one attempt), fused kernels + tok/s on the transformers path, 4B vs 2B acceptance"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "10. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 3, 6, 7, 8 back and upload data/bench/cast_sheet.jpg."
