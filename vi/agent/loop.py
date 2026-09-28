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
    "count_entities": "Distinct confirmed people in a window. args: t_start_ms, t_end_ms (omit for whole episode)",
    "entities_present": "Entities seen for at least min_coverage of the episode ('stayed the whole time'). args: min_coverage (default 0.9)",
    "coverage": "Per-entity seen interval and coverage fraction. args: none",
    "activities": "Activity timeline (what each person did, step by step, objects handled, objects within reach, where they looked). args: entity_id (optional), t_start_ms, t_end_ms",
    "scene": "Latest object inventory per camera view (objects, where, state, counts). args: camera_id (optional)",
    "periods": "The generator's record per 5-second period (people's actions, objects and changes, events, summary). args: t_start_ms, t_end_ms",
    "inspect": "SLOW: look at a person's latest keyframe (or the live frame) with a specific question the script cannot answer "
               "(what exactly they hold, how they do something, where they look). args: question, entity_id (optional)",
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip", "count_entities", "entities_present", "coverage", "activities", "scene", "periods", "inspect"]
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
(4) if the question is ambiguous (which person, which time), ask one clarifying question — but when the cast has exactly one
person, "he", "the man", "the guy" IS that person: answer, do not ask;
(5) answer over entities, not tubes; a person may have several tubes; (6) times are episode-relative
mm:ss.s in scripts and absolute milliseconds in tool args; (7) any count, duration or "whole time" claim MUST come from
count_entities / entities_present / coverage, never from reading the script. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Count people from CONFIRMED entities; BRIEF SIGHTINGS are not people.
Keep answers under 60 words and cite at most 8 ids; cite entity ids (cam1:E7), not tube ids, unless asked about tubes; mention only events
that appear in the script or tool results.
EVENT_TYPES: """ + ", ".join(EVENT_TYPES) + "\nTools: " + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+|W\d{1,3})\b")


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
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):        # never AutoModel: it cannot generate
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
        if warmup:   # CUDA context, kernel compilation (fused linear attention compiles per shape) and cache
            try:     # allocation happen here at a realistic prompt length, not inside the first question
                filler = "SCENE SCRIPT: " + ("cam1:E1 person unnamed seen 00:00.0–00:16.7 coverage 99%; " * 80)
                self.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": filler + "\nQuestion: how many people? Reply with {}"}], {})
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


WHOLE_TIME_RE = re.compile(r"whole (time|episode|clip)|entire (time|episode|clip)|throughout", re.I)
COUNT_Q_RE = re.compile(r"how many|number of (people|persons|workers)|count", re.I)
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}


def _numbers_in(text: str) -> list[int]:
    t = ID_RE.sub(" ", text)
    t = re.sub(r"\d{2}:\d{2}(\.\d)?", " ", t)                        # timestamps are not counts
    nums = [int(m) for m in re.findall(r"(?<![\d.:])(\d{1,3})(?![\d.:])", t)]
    nums += [v for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", t, re.I)]
    return nums


def numeric_claim_issue(engine: Engine, episode_id: str, question: str, step: AnswerStep) -> dict | None:
    """E-AGT-06 for numbers: a people count or a 'stayed the whole time' list in the answer is
    compared with the deterministic tools; a disagreement is sent back once with the true values."""
    try:
        if COUNT_Q_RE.search(question):
            truth = T.count_entities(engine, episode_id)["count"]
            nums = _numbers_in(step.text)
            if nums and truth not in nums and all(abs(n - truth) > 1 for n in nums[:3]):
                return {"msg": f"count_entities says {truth} confirmed people in this episode (your answer implies {nums[:3]})."}
        if WHOLE_TIME_RE.search(question):
            ep = T.entities_present(engine, episode_id, 0.9)
            truth_ids = set(ep["entity_ids"])
            cited = {c for c in step.citations if re.search(r":E\d+$", c)}
            if truth_ids and len(cited & truth_ids) < max(1, len(truth_ids) - 1):
                return {"msg": f"entities_present(min_coverage=0.9) says {len(truth_ids)} entities stayed the whole time: "
                               f"{sorted(truth_ids)}; your answer names {sorted(cited & truth_ids) or 'none of them'}."}
    except Exception:
        return None
    return None


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


EPISODE_TOOLS = {"count_entities", "entities_present", "coverage"}


def run_tool(engine: Engine, step: ToolStep, episode_id: str | None = None) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip, "count_entities": T.count_entities,
          "entities_present": T.entities_present, "coverage": T.coverage}[step.tool]
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
    if step.tool in EPISODE_TOOLS:
        result = fn(engine, episode_id or step.args.get("episode_id", ""), **args)
        if dropped:
            result = {**result, "note": f"ignored unknown args {dropped}"}
        return result
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
                result = run_tool(engine, step, episode_id)
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
        num_issue = numeric_claim_issue(engine, episode_id, question, step)
        if num_issue and not already_revised:
            trace.append({"step": i, "revise": "numeric claim disagrees with tools", "detail": num_issue["msg"]})
            messages.append({"role": "user", "content": "TOOL RESULT: " + num_issue["msg"] + " Answer again using these numbers."})
            continue
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
                 "confidence": step.confidence, "cited": bool(valid), "unsupported_event_claims": bad_claims,
                 "numeric_issue": num_issue["msg"] if num_issue else None}
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


# ---------------------------------------------------------------- multi-episode, live-footage entry point
WINDOW_TOOLS = {"count_entities", "entities_present", "coverage"}

SYSTEM_LIVE_SUFFIX = """
You are answering about recorded footage. FOOTAGE: {start}–{end} ({tz}); the latest processed moment is {now}.
The scene script below covers only the window {ws}–{we}. If the question needs a different time, say which
window you are answering about. PERIODS are the generator's records of each few-second period: what each person did (in
order), the objects they handled and their state, what changed in the scene, events; the latest period's OBJECTS line is the
current inventory. Use them first. SCENE lists the objects in each camera view (the space's inventory: use it for "what is on the
desk / what objects are around"). ACTIVITY lines say what each person was doing, step by step, the objects handled and
their state, counts, what was within reach and where they looked: use them for "what is he doing / how / with what". If the question
needs detail the lines do not have, call inspect with the question and the W-id. W-ids (W1, W2…) are people; a person seen on several cameras has one W-id and several
camera tracks (cam03:E1 …). Count people by W-ids, never by camera tracks. A track id (person:E5, cam02:E1) is a PERSON's
track on a camera, never a place: never write "at person:E1" or "moves between E2 and E3". Places are zones and cameras
and cameras. A track whose looks say NOT A PERSON is a false detection: ignore it. When the cast says "hard evidence: at most N seen
at once", the footage proves no more than N people were ever visible together; several W-ids with the same looks are
probably one person seen from different angles, and you should say so. If nothing in the window matches, say so plainly. Never guess names: people are
unnamed unless the cast says otherwise; describe them from their `looks` instead. Counts carry an uncertainty of
about ±1 person; say "about" for counts above 3. Do not speculate about what happened outside the footage.
"""


CAM_RE = re.compile(r"\b(?:cam(?:era)?)\s*-?\s*(\d{1,2})\b", re.I)


def ground_camera(question: str, cameras: list[str]) -> str | None:
    """'on cam 4' / 'camera 04' / 'CAM 12' -> the store's camera id with that number, if any."""
    m = CAM_RE.search(question)
    if not m or not cameras:
        return None
    n = int(m.group(1))
    for c in cameras:
        digits = re.sub(r"\D", "", c)
        if digits and int(digits) == n:
            return c
    return None


def _window_tool(engine: Engine, step: ToolStep, ws: int, we: int, camera_id: str | None = None) -> Any:
    args = {k: v for k, v in step.args.items() if v is not None}
    a = int(args.get("t_start_ms", ws)); b = int(args.get("t_end_ms", we))
    a, b = max(ws, min(a, b)), min(we, max(a, b))
    if step.tool == "count_entities":
        return T.count_entities_window(engine, a, b, camera_id=camera_id)
    if step.tool == "entities_present":
        return T.entities_present_window(engine, a, b, float(args.get("min_coverage", 0.9)), camera_id=camera_id)
    if step.tool == "coverage":
        return T.coverage_window(engine, a, b, camera_id=camera_id)
    if step.tool == "periods":
        return T.volumes_window(engine, a, b, camera_id)
    if step.tool == "scene":
        return T.scene_inventory(engine, we, args.get("camera_id") or camera_id)
    if step.tool == "activities":
        eid = args.get("entity_id")
        return T.activities_window(engine, a, b, entity_id=eid if eid and not str(eid).startswith("W") else None, camera_id=camera_id)
    return run_tool(engine, step, None)


AFFIRM_RE = re.compile(r"^\s*(?:(?:yes|yeah|yep|yup|exactly|correct|right|that one|that's right|ok|okay|sure|go ahead|please|do it|the first|the second|the latter|the former)[\s,!.]*)+$", re.I)


def resolve_followup(question: str, history: list[dict] | None) -> tuple[str, str | None]:
    """history: [{"q": ..., "a": ..., "action": "answer"|"clarify"}, ...] (most recent last).
    A bare confirmation after a clarification re-asks the previous question with the assistant's
    proposed reading; any other question carries the previous exchange as context."""
    if not history:
        return question, None
    last = history[-1]
    if AFFIRM_RE.match(question) and last.get("action") == "clarify":
        merged = f"{last['q']} (the user confirmed: {last['a']})"
        return merged, f"Previous question: {last['q']}\nYou asked: {last['a']}\nUser: {question}"
    ctx = f"Previous question: {last['q']}\nPrevious answer: {last['a'][:300]}"
    return question, ctx


def ask_window(engine: Engine, question: str, backend, now_ms: int, tz_name: str = "UTC", max_steps: int = 6,
               history: list[dict] | None = None, inspector=None) -> dict:
    """Live-footage question answering: scope check -> time grounding -> window script -> model loop
    with window-aware numeric tools. Refusals and clarifications happen before any model call."""
    t_start_total = time.perf_counter()
    question, context = resolve_followup(question, history)
    bounds = T.footage_bounds(engine)
    if bounds is None:
        return {"question": question, "final": {"action": "answer", "text": "No footage has been processed yet.", "citations": [], "cited": False},
                "trace": [], "steps": 0, "latency": {"total_ms": 0}}
    start_ms, end_ms = bounds
    end_ms = max(end_ms, now_ms if now_ms else end_ms)
    from .scope import classify
    from .timeground import fmt, ground
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    kind, msg = classify(question)
    if kind != "ok":
        return {"question": question, "grounding": kind, "steps": 0, "trace": [],
                "final": {"action": "answer", "text": msg, "citations": [], "cited": True, "handled_by": "scope"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    g = ground(question, now_ms or end_ms, start_ms, end_ms, tz_name)
    if g.kind in ("future", "before_start"):
        return {"question": question, "grounding": g.kind, "steps": 0, "trace": [],
                "final": {"action": "answer", "text": g.message + f" I can answer about {fmt(start_ms, tz)}–{fmt(end_ms, tz)}.",
                          "citations": [], "cited": True, "handled_by": "time"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    ws, we = (g.t_start_ms or start_ms), (g.t_end_ms or end_ms)
    cameras = T.cameras_in_store(engine)
    cam = ground_camera(question, cameras)
    if CAM_RE.search(question) and cam is None and cameras:
        return {"question": question, "grounding": "unknown_camera", "steps": 0, "trace": [],
                "final": {"action": "answer", "text": f"There is no camera with that number. Cameras in the footage: {', '.join(cameras)}.",
                          "citations": [], "cited": True, "handled_by": "scope"},
                "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000)}}
    script = T.window_script(engine, ws, we, tz_name, camera_id=cam)
    seen_ids: set[str] = set(ID_RE.findall(script))
    system = SYSTEM + SYSTEM_LIVE_SUFFIX.format(start=fmt(start_ms, tz), end=fmt(end_ms, tz), tz=tz_name,
                                                 now=fmt(now_ms or end_ms, tz), ws=fmt(ws, tz), we=fmt(we, tz))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"SCENE SCRIPT:\n{script}\n\n" + (f"CONTEXT:\n{context}\n\n" if context else "") + f"Question: {question}"}]
    trace: list[dict] = []
    final: dict | None = None
    model_ms = tool_ms = 0.0
    for i in range(max_steps):
        t0 = time.perf_counter()
        raw = backend.complete(messages, STEP_SCHEMA)
        model_ms += (time.perf_counter() - t0) * 1000
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            step = _salvage(raw)
            if step is None:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"TOOL RESULT: invalid step ({e.errors()[0]['msg']}); reply with one valid JSON object."}]
                trace.append({"step": i, "invalid": raw[:200], "reason": e.errors()[0]["msg"]}); continue
            trace.append({"step": i, "salvaged": e.errors()[0]["msg"]})
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            t0 = time.perf_counter()
            try:
                if step.tool == "inspect":
                    if inspector is None:
                        result = {"answer": "inspect is not available in this deployment"}
                    else:
                        args = {k: v for k, v in step.args.items() if v is not None}
                        eid = args.get("entity_id")
                        if eid and eid.startswith("W"):           # a W-id: inspect its longest track
                            members = T.world_members(engine, ws, we, cam).get(eid, [])
                            eid = members[0] if members else None
                        result = inspector(str(args.get("question", question)), eid, cam, None)
                else:
                    result = _window_tool(engine, step, ws, we, cam)
                text = _compact(result); err = None
            except Exception as e:
                result, text, err = None, f"error: {type(e).__name__}: {e}", str(e)
            tool_ms += (time.perf_counter() - t0) * 1000
            seen_ids.update(ID_RE.findall(text))
            n = len(result) if isinstance(result, list) else (1 if result else 0)
            trace.append({"step": i, "tool": step.tool, "args": step.args, "results": n, "error": err})
            messages.append({"role": "user", "content": f"TOOL RESULT ({step.tool}, {n} item(s)):\n{text}"}); continue
        if isinstance(step, ClarifyStep):
            final = {"action": "clarify", "question": step.question}; break
        valid = [c for c in step.citations if c in seen_ids]
        invalid = [c for c in step.citations if c not in seen_ids]
        already = any("revise" in t for t in trace)
        # numeric claims against the window tools
        issue = None
        try:
            if COUNT_Q_RE.search(question):
                truth = T.count_entities_window(engine, ws, we, camera_id=cam)["count"]; nums = _numbers_in(step.text)
                if nums and truth not in nums and all(abs(n - truth) > 1 for n in nums[:3]):
                    issue = f"count_entities says {truth} confirmed people in this window (your answer implies {nums[:3]})."
            if WHOLE_TIME_RE.search(question):
                ep = T.entities_present_window(engine, ws, we, 0.9, camera_id=cam); truth_ids = set(ep["entity_ids"])
                cited = {c for c in step.citations if re.search(r":E\d+$", c)}
                if truth_ids and len(cited & truth_ids) < max(1, len(truth_ids) - 1):
                    issue = f"entities_present(0.9) says {sorted(truth_ids)} stayed the whole window; you named {sorted(cited & truth_ids) or 'none'}."
        except Exception:
            issue = None
        bad_claims = contradicted_event_claims_window(engine, ws, we, step.text)
        if (issue or bad_claims) and not already:
            m = issue or f"your answer mentions {bad_claims} but no such events exist in this window."
            trace.append({"step": i, "revise": m}); messages.append({"role": "user", "content": "TOOL RESULT: " + m + " Answer again."}); continue
        if invalid and not valid and not already:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in the script or tool results; cite only ids you were shown."}); continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid, "confidence": step.confidence,
                 "cited": bool(valid), "unsupported_event_claims": bad_claims, "numeric_issue": issue}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False, "confidence": 0.0}
    return {"question": question, "grounding": g.kind, "camera_id": cam, "window_ms": [ws, we], "window": f"{fmt(ws, tz)}–{fmt(we, tz)}" + (f" on {cam}" if cam else ""),
            "backend": getattr(backend, "name", "?"), "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round((time.perf_counter() - t_start_total) * 1000), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1, "first_prompt_chars": len(messages[1]["content"]),
                        "gen_tokens": getattr(backend, "last_gen_tokens", None), "tok_s": round(getattr(backend, "last_tok_s", 0.0), 1) or None}}


def contradicted_event_claims_window(engine: Engine, ws: int, we: int, text: str) -> list[str]:
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
    have = {e["type"] for e in T.search_events(engine, t_start_ms=ws, t_end_ms=we, limit=10000)}
    return sorted(et for et in named if et not in have)
