"""Synthetic H.264/MPEG-4 clips for tests and benches, made in-process with PyAV.
A person-sized bright block walks left to right across a noisy background."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_walk_clip(path: str | Path, seconds: int = 6, fps: int = 10, width: int = 320, height: int = 240,
                    codec: str = "mpeg4", seed: int = 0, light_switch_at_s: float | None = None) -> Path:
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    n = seconds * fps
    for i in range(n):
        base = 110 + (60 if light_switch_at_s is not None and i >= light_switch_at_s * fps else 0)
        f = rng.normal(base, 3, (height, width)).clip(0, 255).astype(np.uint8)
        x = 10 + i * (width - 60) // n
        f[height // 3: height // 3 + height // 2, x: x + 24] = 235
        rgb = np.repeat(f[:, :, None], 3, axis=2)
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path
