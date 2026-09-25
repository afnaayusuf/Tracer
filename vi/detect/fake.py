from __future__ import annotations

import numpy as np

from vi.gate.framediff import FrameDiffGate
from vi.schemas import Box

from .base import Detection


class BrightBlobDetector:
    """CPU stand-in for RF-DETR: 'detects' bright rectangles (the synthetic walker) so the whole
    GPU slice, including ROI packing, heartbeats and merging, can run in tests without a GPU."""

    resolution = 0
    optimized = False

    def __init__(self, threshold: float = 0.1, luma: int = 200, min_side: int = 12, **_):
        self.threshold = threshold
        self.luma = luma
        self.min_side = min_side

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        gray = image_rgb.mean(axis=2) if image_rgb.ndim == 3 else image_rgb
        mask = gray > self.luma
        h, w = mask.shape
        b = 4
        mb = mask[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean(axis=(1, 3)) > 0.5
        dets = []
        for comp in FrameDiffGate._components(mb):
            ii = [c[0] for c in comp]
            jj = [c[1] for c in comp]
            box = Box(x1=min(jj) * b, y1=min(ii) * b, x2=(max(jj) + 1) * b, y2=(max(ii) + 1) * b)
            if box.short_side >= self.min_side:
                dets.append(Detection(box=box, class_label="person", confidence=0.9))
        return dets

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(c) for c in crops]
