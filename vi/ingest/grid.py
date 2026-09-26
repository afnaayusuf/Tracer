"""Multiplexed feeds: one video whose frame is an R×C grid of cameras (the usual NVR export).
Each cell becomes a virtual camera with its own id. Burned-in labels ("CAM 04") sit in a corner
of every cell; that corner is a media zone so the text never becomes a detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridSpec:
    rows: int
    cols: int
    margin_px: int = 0                     # border between cells, if the NVR draws one
    label_corner: str = "top-right"        # where the burned-in camera label is
    label_frac: tuple[float, float] = (0.36, 0.14)   # label box as a fraction of cell width/height

    @classmethod
    def parse(cls, s: str, margin_px: int = 0) -> "GridSpec":
        r, c = s.lower().split("x")
        return cls(rows=int(r), cols=int(c), margin_px=margin_px)

    @property
    def n(self) -> int:
        return self.rows * self.cols


def cell_ids(spec: GridSpec, prefix: str = "cam") -> list[str]:
    return [f"{prefix}{i + 1:02d}" for i in range(spec.n)]


def cell_boxes(spec: GridSpec, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) per cell in frame pixels, row-major."""
    cw, ch = frame_w / spec.cols, frame_h / spec.rows
    m = spec.margin_px
    out = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            x1, y1 = int(round(c * cw)) + m, int(round(r * ch)) + m
            x2, y2 = int(round((c + 1) * cw)) - m, int(round((r + 1) * ch)) - m
            out.append((x1, y1, max(x1 + 2, x2), max(y1 + 2, y2)))
    return out


def split(frame: np.ndarray, spec: GridSpec) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    return [np.ascontiguousarray(frame[y1:y2, x1:x2]) for (x1, y1, x2, y2) in cell_boxes(spec, w, h)]


def label_zone(spec: GridSpec, cell_w: int, cell_h: int, camera_id: str, tile_id: str):
    """Media zone over the burned-in label corner of one cell (E-DET-05)."""
    from vi.events import Zone
    lw, lh = int(cell_w * spec.label_frac[0]), int(cell_h * spec.label_frac[1])
    if spec.label_corner == "top-left":
        poly = [(0, 0), (lw, 0), (lw, lh), (0, lh)]
    elif spec.label_corner == "bottom-right":
        poly = [(cell_w - lw, cell_h - lh), (cell_w, cell_h - lh), (cell_w, cell_h), (cell_w - lw, cell_h)]
    elif spec.label_corner == "bottom-left":
        poly = [(0, cell_h - lh), (lw, cell_h - lh), (lw, cell_h), (0, cell_h)]
    else:
        poly = [(cell_w - lw, 0), (cell_w, 0), (cell_w, lh), (cell_w - lw, lh)]
    return Zone(zone_id="label", camera_id=camera_id, tile_id=tile_id, kind="media", polygon=poly)


def compose(cells: list[np.ndarray], spec: GridSpec) -> np.ndarray:
    """Put annotated cells back into one grid frame (for the live view)."""
    if not cells:
        return np.zeros((2, 2, 3), np.uint8)
    ch = max(c.shape[0] for c in cells); cw = max(c.shape[1] for c in cells)
    out = np.zeros((spec.rows * ch, spec.cols * cw, 3), np.uint8)
    for i, c in enumerate(cells[: spec.n]):
        r, k = divmod(i, spec.cols)
        out[r * ch: r * ch + c.shape[0], k * cw: k * cw + c.shape[1]] = c
    return out
