"""Ring 3b, the writer branch: pack person crops into one numbered grid, ask the VLM once, get one
JSON record per cell under the Attributes schema (R22). Runs at tube events, never per frame.
The VLM never sees a whole frame here; it sees crops with cell numbers, so it cannot confuse
people across cells without saying which cell it means."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from vi.schemas import Attributes, Modality
from vi.schemas.contact_sheet import CellResult, ContactSheetResult

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "multicolor"]

WRITER_PROMPT = (
    "This image is a grid of cells; each cell shows one person cropped from a camera, with the cell number in the "
    "yellow strip under it (the strip is a label, not part of the scene). For EVERY cell, describe that person only. "
    "Reply with a JSON array, one object per cell, no prose:\n"
    '[{"cell_id": 0, "is_person": true or false (false if the cell shows no REAL person: a box, a chair, a wall, a reflection, '
    'a mannequin, or a person PRINTED on a poster, packaging or a screen), '
    '"top_color": colour of the most visible upper-body garment (a vest counts), one of ' + str(COLORS) + ' or null, '
    '"bottom_color": same or null, '
    '"headwear": short text or null, "carried_item": short text or null, "role": short text or null, '
    '"description": at most 12 words, "confidence": 0-1}]\n'
    "Use null when a field is not visible. Cells are numbered top-left to bottom-right starting at 0."
)


ACTIVITY_PROMPT = (
    "This image is a grid of cells; each cell shows one person with their surroundings, cropped from a camera, with the "
    "cell number in the yellow strip under it (a label, not part of the scene). For EVERY cell describe what that person is "
    "doing RIGHT NOW. Reply with a JSON array, one object per cell, no prose:\n"
    '[{"cell_id": 0, "is_person": true or false, "activity": "<what they are doing, <= 10 words>", '
    '"objects_nearby": ["<up to 6 noun phrases within their reach>"], '
    '"attention": "<where they are looking, <= 8 words>", "posture": "<one of standing/sitting/bending/walking/lying/reaching>", '
    '"carried_item": short text or null, "confidence": 0-1}]\n'
    "Describe only what is visible; do not guess intentions."
)


SCENE_PROMPT = (
    "This is one full camera view of a workplace. List EVERY distinct object you can see (furniture, tools, boxes, devices, "
    "packaging, products), where it is in the view, and its state. Reply with a JSON array, no prose, at most 25 items:\n"
    '[{"object": short noun phrase (e.g. "cardboard box", "keyboard", "roll of plastic wrap"), "where": short location in the view '
    '(e.g. "on the desk, right", "floor, front-left"), "state": short state or null (e.g. "open", "empty", "sealed", "in use"), '
    '"count": integer if several identical, "confidence": 0-1}]\n'
    "Do not list people. Do not guess what is inside closed containers."
)

NARRATE_PROMPT = (
    "The cells of this grid show the SAME person at successive moments (cell #0 first, then #1, #2 …), a few seconds apart. "
    "Narrate step by step what the person does across the cells, naming the objects handled and what happens to them. "
    "Reply with ONE JSON object, no prose:\n"
    '{"steps": ["step 1 (<= 14 words)", "step 2", ...], "objects_handled": [{"object": noun phrase, "state": short state or null}], '
    '"summary": one sentence (<= 25 words), "counts": {"<object>": integer estimate} or {}, "confidence": 0-1}\n'
    "Describe only what is visible. If nothing changes across cells, say so in the summary."
)


def parse_scene_reply(text: str) -> list[dict]:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("["), t.rfind("]")
    if a < 0 or b < 0:
        return []
    try:
        items = json.loads(t[a:b + 1])
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and it.get("object"):
            cnt = it.get("count")
            out.append({"object": str(it["object"])[:40], "where": (str(it["where"])[:40] if it.get("where") else None),
                        "state": (str(it["state"])[:30] if it.get("state") else None),
                        "count": int(cnt) if isinstance(cnt, (int, float)) and cnt > 0 else None,
                        "confidence": float(it["confidence"]) if isinstance(it.get("confidence"), (int, float)) else 0.6})
    return out[:25]


def parse_narration_reply(text: str) -> dict | None:
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
    steps = [str(x)[:100] for x in (o.get("steps") or []) if isinstance(x, str)][:8]
    objs = []
    for x in (o.get("objects_handled") or []):
        if isinstance(x, dict) and x.get("object"):
            objs.append({"object": str(x["object"])[:40], "state": (str(x["state"])[:30] if x.get("state") else None)})
        elif isinstance(x, str):
            objs.append({"object": x[:40], "state": None})
    counts = {str(k)[:40]: int(v) for k, v in (o.get("counts") or {}).items() if isinstance(v, (int, float))} if isinstance(o.get("counts"), dict) else {}
    conf = o.get("confidence", 0.6)
    return {"steps": steps, "objects_handled": objs[:8], "summary": str(o.get("summary") or "")[:200], "counts": counts,
            "confidence": float(conf) if isinstance(conf, (int, float)) else 0.6}


def pack_sheet(crops: list[np.ndarray], cell: int = 224, cols: int = 4, caption: int = 22):
    """Grid with hard borders and a number in a caption strip BELOW each crop, never over it (a badge
    on top of a 30-px crop is what the model ends up describing). Small crops are upscaled to fill
    the cell so the model sees a person, not a speck (E-FOV-04)."""
    from PIL import Image, ImageDraw
    n = len(crops)
    cols = min(cols, max(1, n))
    rows = (n + cols - 1) // cols
    ch = cell + caption
    sheet = Image.new("RGB", (cols * cell, rows * ch), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(crops):
        x, y = (i % cols) * cell, (i // cols) * ch
        im = Image.fromarray(np.ascontiguousarray(c)).convert("RGB")
        scale = min((cell - 8) / max(1, im.width), (cell - 8) / max(1, im.height))   # up or down to fill the cell
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        sheet.paste(im, (x + 4 + (cell - 8 - im.width) // 2, y + 4 + (cell - 8 - im.height) // 2))
        dr.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=(255, 255, 255), width=2)
        dr.rectangle([x, y + cell, x + cell - 1, y + ch - 1], fill=(255, 215, 0))
        dr.text((x + 6, y + cell + 4), f"cell #{i}", fill=(0, 0, 0))
    return sheet


def _color(v):
    return v.lower() if isinstance(v, str) and v.lower() in COLORS else None


def parse_sheet_reply(text: str, tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
    """Tolerant parse of the VLM reply into the schema; None if it cannot be made valid (E-FOV-07)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end < 0:
        return None
    try:
        items = json.loads(t[start:end + 1])
    except Exception:
        return None
    cells: list[CellResult] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            cid = int(it.get("cell_id"))
        except Exception:
            continue
        if not (0 <= cid < len(tube_ids)):
            continue
        desc_bits = [str(it.get("description") or "")[:120]]
        if it.get("is_person") is False:
            desc_bits.insert(0, "NOT A PERSON")
        if it.get("headwear"):
            desc_bits.append(f"headwear: {str(it['headwear'])[:40]}")
        if it.get("role"):
            desc_bits.append(f"role: {str(it['role'])[:30]}")
        conf = it.get("confidence", 0.6)
        conf = float(conf) if isinstance(conf, (int, float)) else 0.6
        objs = it.get("objects_nearby") or []
        objs = [str(o)[:30] for o in objs if isinstance(o, (str, int))][:6] if isinstance(objs, list) else []
        kwargs = dict(modality=modality, carried_item=(str(it["carried_item"])[:60] if it.get("carried_item") else None),
                      carried_item_confidence=0.6 if it.get("carried_item") else 0.0,
                      description="; ".join(b for b in desc_bits if b)[:240], confidence=max(0.0, min(1.0, conf)),
                      activity=(str(it["activity"])[:80] if it.get("activity") else None), objects_nearby=objs,
                      attention=(str(it["attention"])[:60] if it.get("attention") else None),
                      posture=(str(it["posture"])[:30] if it.get("posture") else None))
        if modality.has_color:
            kwargs.update(top_color=_color(it.get("top_color")), bottom_color=_color(it.get("bottom_color")))
        else:
            kwargs.update(color_reason="ir_mode")
        try:
            cells.append(CellResult(cell_id=cid, tube_id=tube_ids[cid], attributes=Attributes(**kwargs)))
        except Exception:
            continue
    # fill missing cells with an empty attributes record so the sheet validates (per-cell failure, not sheet failure)
    have = {c.cell_id for c in cells}
    for i, tid in enumerate(tube_ids):
        if i not in have:
            cells.append(CellResult(cell_id=i, tube_id=tid, attributes=Attributes(
                modality=modality, confidence=0.0, **({} if modality.has_color else {"color_reason": "ir_mode"}))))
    cells.sort(key=lambda c: c.cell_id)
    try:
        return ContactSheetResult(sheet_id=f"sheet_{int(time.time() * 1000)}", modality=modality,
                                  expected_cells=len(tube_ids), cells=cells)
    except Exception:
        return None


class WriterVLM:
    """Qwen3.5 (or any transformers image-text model) on a packed sheet. `backend` may be an
    existing TransformersBackend to reuse its loaded model and processor."""

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", backend=None, max_new_tokens: int = 700, device: str | None = None):
        import torch
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        if backend is not None and getattr(backend, "model", None) is not None:
            self.model, self.proc, self.device = backend.model, backend.tok, backend.device
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(model_id)
        self.calls = 0
        self.last_ms = 0.0

    def inspect(self, image: np.ndarray, question: str, max_new_tokens: int = 200) -> str:
        """The slow path: one frame or crop, the user's actual question, plain-text answer."""
        from PIL import Image
        im = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": im},
                    {"type": "text", "text": "Answer the question about this camera frame in <= 40 words, describing only what is visible. Question: " + question}]}]
        t0 = time.perf_counter()
        try:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return re.sub(r"<think>.*?</think>", "", tok.decode(gen, skip_special_tokens=True), flags=re.DOTALL).strip()

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        from PIL import Image
        im = image if isinstance(image, Image.Image) else Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
        t0 = time.perf_counter()
        try:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return tok.decode(gen, skip_special_tokens=True)

    def scene(self, frame: np.ndarray) -> list[dict]:
        """Full-frame inventory: what objects are in this view, where, in what state."""
        return parse_scene_reply(self._generate(frame, SCENE_PROMPT, 900))

    def narrate(self, frames: list[np.ndarray]) -> dict | None:
        """The same person over time: step-by-step narration with the objects handled."""
        sheet = pack_sheet(frames, cell=256, cols=3)
        return parse_narration_reply(self._generate(sheet, NARRATE_PROMPT, 500))

    def delta(self, vol, state: dict, tz_name: str = "UTC") -> dict | None:
        """Stream mode: the last second's frames + the current state -> only the changes, <= 80 new tokens."""
        from .volume import build_delta_prompt, parse_delta_reply
        prompt = build_delta_prompt(len(vol.frames), vol.period_ms / 1000, state)
        rec = self._volume_text(vol, prompt, tz_name, max_new_tokens=80)
        return parse_delta_reply(rec or "")

    def _volume_text(self, vol, prompt: str, tz_name: str, max_new_tokens: int) -> str | None:
        from .volume import grid_fallback, volume_to_messages
        t0 = time.perf_counter()
        text = None
        try:
            messages, _ = volume_to_messages(vol, prompt, tz_name)
            try:
                inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
            except TypeError:
                inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            tok = getattr(self.proc, "tokenizer", self.proc)
            text = tok.decode(gen, skip_special_tokens=True)
            self.video_ok = True
        except Exception as e:
            self.video_ok = False
            self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            text = self._generate(grid_fallback(vol, tz_name), prompt + "\n(The frames are laid out as a grid with their times under each cell.)", max_new_tokens)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return text

    def volume(self, vol, prompt: str, tz_name: str = "UTC", max_new_tokens: int = 700) -> dict | None:
        """One call per frame-volume. Video input if the processor takes it; a labelled time grid otherwise."""
        from .volume import grid_fallback, parse_volume_reply, volume_to_messages
        t0 = time.perf_counter()
        text = None
        try:
            messages, _ = volume_to_messages(vol, prompt, tz_name)
            try:
                inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
            except TypeError:
                inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            tok = getattr(self.proc, "tokenizer", self.proc)
            text = tok.decode(gen, skip_special_tokens=True)
            self.video_ok = True
        except Exception as e:                                   # processor without video support: labelled grid
            self.video_ok = False
            self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            text = self._generate(grid_fallback(vol, tz_name), prompt + "\n(The frames are laid out as a grid with their times under each cell.)", max_new_tokens)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return parse_volume_reply(text or "")

    def describe(self, crops: list[np.ndarray], tube_ids: list[str], modality: Modality = Modality.rgb, mode: str = "appearance") -> ContactSheetResult | None:
        sheet = pack_sheet(crops)
        prompt = ACTIVITY_PROMPT if mode == "activity" else WRITER_PROMPT
        messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": prompt}]}]
        t0 = time.perf_counter()
        try:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        text = tok.decode(gen, skip_special_tokens=True)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return parse_sheet_reply(text, tube_ids, modality)


class OpenAIWriter:
    """The same writer roles against an OpenAI-compatible multimodal server (vLLM / SGLang): frames go as
    base64 data URLs. This is the fast path: continuous batching, prefix caching, 100+ tok/s decode."""

    name = "openai-writer"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen3.5-35B-A3B", api_key: str = "EMPTY", timeout: float = 120.0):
        import urllib.request
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self._req = urllib.request
        self.calls = 0; self.last_ms = 0.0; self.video_ok = True; self.last_error = None

    def _chat(self, images, text: str, max_tokens: int) -> str:
        import base64, io, json as _json
        from PIL import Image
        content = []
        for im in images:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(np.ascontiguousarray(im)).convert("RGB")
            buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=85)
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        req = self._req.Request(self.base_url + "/chat/completions", data=_json.dumps(body).encode(), method="POST",
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with self._req.urlopen(req, timeout=self.timeout) as r:
                out = _json.loads(r.read().decode())
        except Exception:
            body.pop("chat_template_kwargs", None)
            req = self._req.Request(self.base_url + "/chat/completions", data=_json.dumps(body).encode(), method="POST",
                                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            with self._req.urlopen(req, timeout=self.timeout) as r:
                out = _json.loads(r.read().decode())
        self.calls += 1; self.last_ms = (time.perf_counter() - t0) * 1000
        return out["choices"][0]["message"]["content"]

    def describe(self, crops, tube_ids, modality=Modality.rgb, mode: str = "appearance"):
        return parse_sheet_reply(self._chat([pack_sheet(crops)], ACTIVITY_PROMPT if mode == "activity" else WRITER_PROMPT, 700), tube_ids, modality)

    def scene(self, frame):
        return parse_scene_reply(self._chat([frame], SCENE_PROMPT, 900))

    def narrate(self, frames):
        return parse_narration_reply(self._chat([pack_sheet(frames, cell=256, cols=3)], NARRATE_PROMPT, 500))

    def inspect(self, image, question: str, max_new_tokens: int = 200) -> str:
        return re.sub(r"<think>.*?</think>", "", self._chat([image], "Answer the question about this camera frame in <= 40 words, describing only what is visible. Question: " + question, max_new_tokens), flags=re.DOTALL).strip()

    def volume(self, vol, prompt: str, tz_name: str = "UTC", max_new_tokens: int = 700):
        from .volume import parse_volume_reply
        return parse_volume_reply(self._chat(list(vol.frames), prompt + "\n(The images are the frames in time order.)", max_new_tokens))

    def delta(self, vol, state: dict, tz_name: str = "UTC"):
        from .volume import build_delta_prompt, parse_delta_reply
        return parse_delta_reply(self._chat(list(vol.frames), build_delta_prompt(len(vol.frames), vol.period_ms / 1000, state) + "\n(The images are the frames in time order.)", 80))
