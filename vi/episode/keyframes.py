from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from vi.schemas import Box


class KeyframeStore:
    """E-FOV-06 / R20: persist a padded native-resolution crop the moment a tube is born,
    before the ring buffer can expire. Local JPEGs now; the same refs point at R2 later."""

    def __init__(self, root: str | Path, pad: float = 0.2, quality: int = 90):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pad = pad
        self.quality = quality
        self.saved = 0

    def save(self, camera_id: str, t_ms: int, box: Box, frame_rgb: np.ndarray, tag: str = "birth") -> str:
        h, w = frame_rgb.shape[:2]
        px, py = box.width * self.pad, box.height * self.pad
        x1, y1 = int(max(0, box.x1 - px)), int(max(0, box.y1 - py))
        x2, y2 = int(min(w, box.x2 + px)), int(min(h, box.y2 + py))
        crop = frame_rgb[y1:y2, x1:x2]
        rel = Path(camera_id) / f"{t_ms}_{tag}_{x1}_{y1}.jpg"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(np.ascontiguousarray(crop)).save(path, quality=self.quality)
        except ImportError:  # keep going without Pillow: raw .npy
            path = path.with_suffix(".npy")
            np.save(path, crop)
        self.saved += 1
        return f"kf://{rel.with_suffix(path.suffix).as_posix()}"

    def make_sink(self, current_frame: Callable[[], np.ndarray | None]) -> Callable[[str, int, Box], str]:
        """Adapter for SimpleIoUTracker(keyframe_sink=...): the tracker only knows the box,
        the closure knows the frame."""

        def sink(camera_id: str, t_ms: int, box: Box) -> str:
            frame = current_frame()
            if frame is None:
                return f"kf://missing/{camera_id}/{t_ms}"
            return self.save(camera_id, t_ms, box, frame)

        return sink
