from __future__ import annotations

import numpy as np

from vi.schemas import Box

from .base import Detection

COCO_KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat",
             "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup", "laptop",
             "cell phone", "chair", "couch", "bed", "dining table", "tv", "potted plant"}


class RFDETRDetector:
    """Ring 1 on GPU. rfdetr==1.7.0 (Apache 2.0 for nano..large; XL/2XL excluded, R12).
    Import is lazy so the CPU slice and the tests never need torch."""

    def __init__(self, size: str = "nano", threshold: float = 0.4, device: str = "cuda"):
        try:
            import rfdetr  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install rfdetr==1.7.0 (GPU runtime); not needed for the CPU slice") from e
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall  # type: ignore
        cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}[size]
        self.model = cls(device=device)
        self.threshold = threshold
        self.names = getattr(self.model, "class_names", None)

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        results = self.model.predict(crops, threshold=self.threshold)
        if not isinstance(results, list):
            results = [results]
        for r in results:
            dets = []
            for xyxy, cid, conf in zip(r.xyxy, r.class_id, r.confidence):
                label = self.names[int(cid)] if self.names else str(int(cid))
                if self.names and label not in COCO_KEEP:
                    continue
                dets.append(Detection(box=Box(x1=float(xyxy[0]), y1=float(xyxy[1]),
                                              x2=float(xyxy[2]), y2=float(xyxy[3])),
                                      class_label=label, confidence=float(conf)))
            out.append(dets)
        return out
