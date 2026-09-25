from __future__ import annotations

import numpy as np

from vi.schemas import Box

from .base import Detection

# COCO classes the engine keeps as tube candidates; everything else is dropped at the detector.
KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "backpack",
        "umbrella", "handbag", "suitcase", "bottle", "cup", "laptop", "cell phone", "chair",
        "couch", "bed", "dining table", "tv", "potted plant", "book", "clock", "vase", "scissors"}

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}
DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576, "large": 704}


class RFDETRDetector:
    """Ring 1 on GPU. rfdetr==1.7.0, Apache-2.0 for nano..large (R12). Verified against the
    1.7.0 source: predict() takes one RGB ndarray or a list of them and returns
    supervision Detections with class names in data["class_name"]; optimize_for_inference()
    traces the model at one fixed batch size, hence pad_batch() in roi.py."""

    def __init__(self, size: str = "nano", threshold: float = 0.5, batch_size: int = 1,
                 fp16: bool = True, optimize: bool = True, keep: set[str] | None = None):
        try:
            import rfdetr  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install rfdetr==1.7.0 (GPU runtime); not needed for the CPU harness") from e
        import rfdetr as _rf
        self.size = size
        self.threshold = threshold
        self.batch_size = batch_size
        self.keep = KEEP if keep is None else keep
        self.model = getattr(_rf, SIZES[size])()
        self.optimized = False
        if optimize:
            try:
                self.model.optimize_for_inference(compile=True, batch_size=batch_size,
                                                  dtype="float16" if fp16 else "float32")
                self.optimized = True
            except Exception as e:  # tracing can fail on exotic setups; fall back to eager
                print(f"[rfdetr] optimize_for_inference failed ({e!r}); running eager")
        self.resolution = DEFAULT_RESOLUTION[size]

    def _to_detections(self, r) -> list[Detection]:
        dets: list[Detection] = []
        if r is None or r.xyxy is None or len(r.xyxy) == 0:
            return dets
        names = r.data.get("class_name") if getattr(r, "data", None) else None
        for i, xyxy in enumerate(r.xyxy):
            label = str(names[i]) if names is not None else str(int(r.class_id[i]))
            if self.keep and label not in self.keep:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), class_label=label,
                                  confidence=float(r.confidence[i])))
        return dets

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        return self.detect_batch([image_rgb])[0]

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        """The traced model only accepts exactly `batch_size` images, so every call is padded
        to that size (repeating the last image) and the padding results are dropped. Callers
        may pass 1..batch_size images; longer lists are processed in chunks."""
        if not crops:
            return []
        out: list[list[Detection]] = []
        for i in range(0, len(crops), self.batch_size):
            chunk = list(crops[i:i + self.batch_size])
            real = len(chunk)
            if self.optimized:
                while len(chunk) < self.batch_size:
                    chunk.append(chunk[-1])
            results = self.model.predict(chunk if len(chunk) > 1 else chunk[0], threshold=self.threshold,
                                         include_source_image=False)
            if not isinstance(results, list):
                results = [results]
            out += [self._to_detections(r) for r in results[:real]]
        return out
