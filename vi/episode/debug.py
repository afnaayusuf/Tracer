from __future__ import annotations

from pathlib import Path

import numpy as np

from vi.detect import ROI, Detection
from vi.schemas import Tube

COL = {"roi": (60, 200, 60), "full": (60, 160, 255), "det_roi": (255, 220, 40), "det_full": (80, 230, 255),
       "active": (255, 60, 60), "occluded": (255, 140, 40), "born": (200, 200, 200)}


def annotate(frame_rgb: np.ndarray, rois: list[ROI], dets: list[Detection], tubes: list[Tube],
             title: str, path: str | Path) -> Path | None:
    """Debug frame: crops (green; full frame blue), detections (yellow from crops, cyan from the
    full frame, dashed-ish by confidence label), tubes (red active, orange predicted, grey born)
    with id suffix and state. Opens in Colab's file browser; upload one to the chat to review."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    im = Image.fromarray(np.ascontiguousarray(frame_rgb))
    dr = ImageDraw.Draw(im)
    for r in rois:
        c = COL["full"] if r.source_blobs == 0 else COL["roi"]
        dr.rectangle([r.x1, r.y1, r.x2 - 1, r.y2 - 1], outline=c, width=1)
    for d in dets:
        c = COL["det_full"] if d.origin == "full" else COL["det_roi"]
        b = d.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=1)
        dr.text((b.x1 + 2, b.y2 - 12), f"{d.class_label[:6]} {d.confidence:.2f}{'t' if d.roi_truncated else ''}", fill=c)
    for t in tubes:
        c = COL.get(t.state.value, COL["born"])
        b = t.box
        dr.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=3)
        dr.text((b.x1 + 2, max(0, b.y1 - 12)), f"#{t.tube_id.split(':')[-1]} {t.state.value[:3]}", fill=c)
    dr.text((6, 6), title, fill=(255, 255, 255))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=85)
    return path
