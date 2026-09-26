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
