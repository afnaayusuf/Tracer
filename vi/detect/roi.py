from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vi.gate.base import MotionBlob
from vi.schemas import Box

from .base import Detection

EDGE_PX = 2


class ROI(BaseModel):
    """A crop window in native frame coordinates (ints, inclusive-exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int
    source_blobs: int = 1

    @property
    def box(self) -> Box:
        return Box(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


def blobs_to_rois(blobs: list[MotionBlob], frame_w: int, frame_h: int, stride: int = 1,
                  pad: float = 0.25, min_side: int = 96, merge_iou: float = 0.05) -> list[ROI]:
    """R12: turn gate blobs (in gate coordinates, downscaled by `stride`) into padded,
    clamped, merged crop windows in native coordinates. Overlapping windows are merged so
    one object never yields two crops; tiny windows are grown to `min_side` so the
    detector sees enough context."""
    boxes: list[Box] = []
    for b in blobs:
        x1, y1, x2, y2 = b.box.x1 * stride, b.box.y1 * stride, b.box.x2 * stride, b.box.y2 * stride
        w, h = x2 - x1, y2 - y1
        px, py = max(w * pad, (min_side - w) / 2, 0), max(h * pad, (min_side - h) / 2, 0)
        boxes.append(Box(x1=max(0, x1 - px), y1=max(0, y1 - py),
                         x2=min(frame_w, x2 + px), y2=min(frame_h, y2 + py)))
    # greedy merge of overlapping windows
    merged: list[tuple[Box, int]] = []
    for bx in sorted(boxes, key=lambda b: -b.area):
        for i, (m, n) in enumerate(merged):
            if m.iou(bx) > merge_iou or _contains(m, bx):
                merged[i] = (Box(x1=min(m.x1, bx.x1), y1=min(m.y1, bx.y1),
                                 x2=max(m.x2, bx.x2), y2=max(m.y2, bx.y2)), n + 1)
                break
        else:
            merged.append((bx, 1))
    return [ROI(x1=int(m.x1), y1=int(m.y1), x2=int(np.ceil(m.x2)), y2=int(np.ceil(m.y2)), source_blobs=n)
            for m, n in merged if m.x2 - m.x1 >= 8 and m.y2 - m.y1 >= 8]


def _contains(outer: Box, inner: Box) -> bool:
    return outer.x1 <= inner.x1 and outer.y1 <= inner.y1 and outer.x2 >= inner.x2 and outer.y2 >= inner.y2


def crop_roi(frame_rgb: np.ndarray, roi: ROI) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[roi.y1:roi.y2, roi.x1:roi.x2])


def remap_detections(dets: list[Detection], roi: ROI, frame_w: int, frame_h: int) -> list[Detection]:
    """Shift crop-space boxes back to frame space and flag boxes touching the frame border
    (E-DET-02): a person cut off at the edge has an unreliable foot point."""
    out = []
    for d in dets:
        b = Box(x1=d.box.x1 + roi.x1, y1=d.box.y1 + roi.y1, x2=d.box.x2 + roi.x1, y2=d.box.y2 + roi.y1)
        truncated = b.x1 <= EDGE_PX or b.y1 <= EDGE_PX or b.x2 >= frame_w - EDGE_PX or b.y2 >= frame_h - EDGE_PX
        out.append(d.model_copy(update={"box": b, "truncated": truncated}))
    return out


def pad_batch(crops: list[np.ndarray], batch_size: int) -> tuple[list[np.ndarray], int]:
    """A traced model runs at one fixed batch size; pad with the last crop and report how
    many entries are real so the caller drops the padding."""
    if not crops:
        return [], 0
    real = len(crops)
    padded = list(crops[:batch_size])
    while len(padded) < batch_size:
        padded.append(padded[-1])
    return padded, min(real, batch_size)
