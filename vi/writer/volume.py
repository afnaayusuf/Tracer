"""Frame volumes: the generator's unit of perception (R-volume). A period P of footage is sampled
into K frames with timestamps and given to the VLM as a video, so its 3-D position encoding
(time, height, width) sees the volume. The tracker's ids are drawn on the frames so identity is
stable across volumes; the model does everything semantic in ONE call per volume.

The prompt is schema-only on purpose: format placeholders, no scene vocabulary. Words about the
scene must come from the footage, never from the person who wrote the prompt."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

VOLUME_PROMPT = (
    "You are given {k} frames of the same camera view covering {period:.0f} seconds, in time order, with their timestamps. "
    "People carry an id label drawn on their box; use those ids. Report what happened in this period as ONE JSON object, no prose:\n"
    "{{\n"
    '  "people": [{{"id": "<id label on the box>", "appearance": "<short, visible clothing/features>", '
    '"actions": ["<step, <= 14 words, in order>", "..."], "objects_handled": [{{"object": "<noun phrase>", "state": "<visible state or null>"}}], '
    '"attention": "<where they look, <= 8 words>", "posture": "<one word>", "location": "<where in the view>"}}],\n'
    '  "objects": [{{"object": "<noun phrase>", "where": "<where in the view>", "state": "<visible state or null>", "count": <integer or null>, '
    '"changed": <true if it moved, appeared, disappeared or changed state in this period>}}],\n'
    '  "events": [{{"t": "<timestamp of one of the frames>", "who": "<id or null>", "what": "<<= 12 words>", "kind": "<one word category>"}}],\n'
    '  "summary": "<one sentence, <= 30 words>",\n'
    '  "unclear": ["<anything you could not determine, <= 10 words each>"],\n'
    '  "confidence": <0-1>\n'
    "}}\n"
    "Rules: describe only what is visible in these frames; do not infer intentions; if a person has no id label, use \"unlabeled\"; "
    "do not list people as objects; a printed image or a reflection of a person is not a person.{context}"
)


@dataclass
class FrameVolume:
    camera_id: str                     # a camera, or a tile id for a composite view
    t0_ms: int
    period_ms: int
    frames: list                       # np.ndarray RGB, annotated (boxes + ids drawn)
    times_ms: list[int]
    ids_present: list[str] = field(default_factory=list)


def sample_times(t0_ms: int, period_ms: int, k: int) -> list[int]:
    """t_i = t0 + (i + 1/2) * P / K: centred samples, never the exact boundary frame twice."""
    return [int(t0_ms + (i + 0.5) * period_ms / k) for i in range(k)]


def downscale(frame: np.ndarray, max_w: int) -> np.ndarray:
    from PIL import Image
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    s = max_w / w
    return np.asarray(Image.fromarray(np.ascontiguousarray(frame)).resize((max_w, max(1, int(h * s))), Image.BILINEAR))


def visual_tokens(w: int, h: int, k: int, patch: int = 28, temporal_merge: int = 2) -> int:
    """Rough token cost of a volume for a Qwen-VL class model: (H/patch)(W/patch) per frame, halved by temporal merging."""
    per_frame = max(1, w // patch) * max(1, h // patch)
    return int(per_frame * k / temporal_merge)


def annotate_ids(frame: np.ndarray, boxes: list[tuple[str, tuple[float, float, float, float]]]) -> np.ndarray:
    """Draw each tracked person's world id on its box so the model can refer to it."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    dr = ImageDraw.Draw(im)
    for label, (x1, y1, x2, y2) in boxes:
        dr.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=3)
        tw = 8 * len(label) + 8
        dr.rectangle([x1, max(0, y1 - 18), x1 + tw, max(18, y1)], fill=(255, 215, 0))
        dr.text((x1 + 4, max(0, y1 - 16)), label, fill=(0, 0, 0))
    return np.asarray(im)


def build_prompt(k: int, period_s: float, previous_summary: str | None = None, known_ids: list[str] | None = None) -> str:
    ctx = ""
    if previous_summary or known_ids:
        ctx = "\nContext from the previous period (may have changed): " + (previous_summary or "") + \
              (f" Ids seen recently: {', '.join(known_ids)}." if known_ids else "")
    return VOLUME_PROMPT.format(k=k, period=period_s, context=ctx)


def parse_volume_reply(text: str) -> dict | None:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        o = json.loads(t[a:b + 1])
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    def s(x, n): return str(x)[:n] if x is not None else None
    people = []
    for p in (o.get("people") or []):
        if not isinstance(p, dict):
            continue
        people.append({"id": s(p.get("id"), 40) or "unlabeled", "appearance": s(p.get("appearance"), 120),
                       "actions": [s(x, 100) for x in (p.get("actions") or []) if isinstance(x, str)][:8],
                       "objects_handled": [{"object": s(x.get("object"), 40), "state": s(x.get("state"), 30)} for x in (p.get("objects_handled") or []) if isinstance(x, dict) and x.get("object")][:8],
                       "attention": s(p.get("attention"), 60), "posture": s(p.get("posture"), 20), "location": s(p.get("location"), 60)})
    objects = []
    for x in (o.get("objects") or []):
        if isinstance(x, dict) and x.get("object"):
            c = x.get("count")
            objects.append({"object": s(x["object"], 40), "where": s(x.get("where"), 40), "state": s(x.get("state"), 30),
                            "count": int(c) if isinstance(c, (int, float)) and c > 0 else None, "changed": bool(x.get("changed", False))})
    events = []
    for e in (o.get("events") or []):
        if isinstance(e, dict) and e.get("what"):
            events.append({"t": s(e.get("t"), 20), "who": s(e.get("who"), 40), "what": s(e["what"], 80), "kind": s(e.get("kind"), 20)})
    conf = o.get("confidence", 0.6)
    return {"people": people[:12], "objects": objects[:30], "events": events[:20], "summary": s(o.get("summary"), 240) or "",
            "unclear": [s(x, 60) for x in (o.get("unclear") or []) if isinstance(x, str)][:6],
            "confidence": float(conf) if isinstance(conf, (int, float)) else 0.6}


def volume_to_messages(vol: FrameVolume, prompt: str, tz_name: str = "UTC"):
    """Native video input when the processor supports it (frames + fps -> temporal position ids);
    the caller falls back to a labelled grid if not."""
    from PIL import Image
    frames = [Image.fromarray(np.ascontiguousarray(f)).convert("RGB") for f in vol.frames]
    fps = max(0.1, len(frames) / max(0.1, vol.period_ms / 1000))
    return [{"role": "user", "content": [{"type": "video", "video": frames, "fps": fps}, {"type": "text", "text": prompt}]}], frames


def grid_fallback(vol: FrameVolume, tz_name: str = "UTC"):
    """Labelled time grid: one cell per frame with its clock time under it."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from PIL import Image, ImageDraw
    tz = ZoneInfo(tz_name)
    n = len(vol.frames); cols = 3 if n > 4 else 2
    rows = (n + cols - 1) // cols
    cw = 448; ch = int(cw * vol.frames[0].shape[0] / max(1, vol.frames[0].shape[1])); cap = 20
    sheet = Image.new("RGB", (cols * cw, rows * (ch + cap)), (20, 20, 20)); dr = ImageDraw.Draw(sheet)
    for i, (f, t) in enumerate(zip(vol.frames, vol.times_ms)):
        x, y = (i % cols) * cw, (i // cols) * (ch + cap)
        sheet.paste(Image.fromarray(np.ascontiguousarray(f)).convert("RGB").resize((cw, ch)), (x, y))
        dr.rectangle([x, y + ch, x + cw - 1, y + ch + cap - 1], fill=(255, 215, 0))
        dr.text((x + 6, y + ch + 3), f"#{i}  {datetime.fromtimestamp(t / 1000, tz).strftime('%H:%M:%S')}", fill=(0, 0, 0))
    return sheet


# ---------------------------------------------------------------- stream mode: one second in, only the changes out
DELTA_PROMPT = (
    "You are given {k} consecutive frames covering the last {period:.0f} second(s) of one camera view, and the CURRENT STATE recorded so far. "
    "People carry an id label drawn on their box; use those ids. Reply with ONE compact JSON object describing ONLY what changed in these "
    "frames compared with the state, in at most 60 tokens, no prose:\n"
    '{{"p": {{"<id>": {{"a": "<new action, <= 8 words>", "o": ["<object handled>"]}}}}, "s": [{{"o": "<object>", "st": "<new state>"}}], '
    '"e": "<event, <= 8 words, or omit>"}}\n'
    "If nothing changed, reply exactly {{}}. Never repeat what the state already says. Describe only what is visible.\n"
    "CURRENT STATE: {state}"
)


def build_delta_prompt(k: int, period_s: float, state: dict) -> str:
    compact = {"people": {pid: {"action": v.get("action"), "objects": v.get("objects", [])[:4]} for pid, v in (state.get("people") or {}).items()},
               "objects": [{"o": o["object"], "st": o.get("state")} for o in (state.get("objects") or [])[:12]]}
    return DELTA_PROMPT.format(k=k, period=period_s, state=json.dumps(compact, separators=(",", ":"))[:900])


def parse_delta_reply(text: str) -> dict | None:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        o = json.loads(t[a:b + 1])
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    people = {}
    for pid, v in (o.get("p") or {}).items():
        if isinstance(v, dict):
            people[str(pid)[:20]] = {"action": (str(v.get("a"))[:80] if v.get("a") else None),
                                     "objects": [str(x)[:40] for x in (v.get("o") or []) if isinstance(x, str)][:6]}
    objects = [{"object": str(x.get("o"))[:40], "state": (str(x.get("st"))[:30] if x.get("st") else None)} for x in (o.get("s") or []) if isinstance(x, dict) and x.get("o")]
    event = str(o.get("e"))[:80] if o.get("e") else None
    return {"people": people, "objects": objects, "event": event, "empty": not (people or objects or event)}


def apply_delta(state: dict, delta: dict, t_ms: int) -> dict:
    """Fold a delta into the running state (what the next prompt sees, and what the lib snapshot is)."""
    st = {"people": dict(state.get("people") or {}), "objects": list(state.get("objects") or []), "t_ms": t_ms}
    for pid, v in (delta.get("people") or {}).items():
        cur = dict(st["people"].get(pid) or {})
        if v.get("action"):
            cur["action"] = v["action"]; cur["since_ms"] = t_ms
        if v.get("objects"):
            cur["objects"] = v["objects"]
        st["people"][pid] = cur
    for o in delta.get("objects") or []:
        for existing in st["objects"]:
            if existing["object"].lower() == o["object"].lower():
                existing["state"] = o["state"]; existing["changed_ms"] = t_ms; break
        else:
            st["objects"].append({"object": o["object"], "state": o["state"], "changed_ms": t_ms})
    st["objects"] = st["objects"][-30:]
    return st
