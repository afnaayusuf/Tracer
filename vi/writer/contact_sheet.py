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
    '[{"cell_id": 0, "is_person": true or false (false if the cell shows no person: a box, a chair, a wall, a reflection), '
    '"top_color": colour of the most visible upper-body garment (a vest counts), one of ' + str(COLORS) + ' or null, '
    '"bottom_color": same or null, '
    '"headwear": short text or null, "carried_item": short text or null, "role": short text or null, '
    '"description": at most 12 words, "confidence": 0-1}]\n'
    "Use null when a field is not visible. Cells are numbered top-left to bottom-right starting at 0."
)


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
        kwargs = dict(modality=modality, carried_item=(str(it["carried_item"])[:60] if it.get("carried_item") else None),
                      carried_item_confidence=0.6 if it.get("carried_item") else 0.0,
                      description="; ".join(b for b in desc_bits if b)[:240], confidence=max(0.0, min(1.0, conf)))
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

    def describe(self, crops: list[np.ndarray], tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
        sheet = pack_sheet(crops)
        messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": WRITER_PROMPT}]}]
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
