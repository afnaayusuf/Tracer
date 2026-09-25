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
