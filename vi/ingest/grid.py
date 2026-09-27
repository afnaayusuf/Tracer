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
    label_corners: tuple[str, ...] = ("top-right", "bottom-left")   # channel name / timestamp overlays (NVRs use both)
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


def label_zones(spec: GridSpec, cell_w: int, cell_h: int, camera_id: str, tile_id: str) -> list:
    """Media zones over the burned-in overlay corners of one cell (E-DET-05)."""
    from vi.events import Zone
    lw, lh = int(cell_w * spec.label_frac[0]), int(cell_h * spec.label_frac[1])
    polys = {"top-left": [(0, 0), (lw, 0), (lw, lh), (0, lh)],
             "top-right": [(cell_w - lw, 0), (cell_w, 0), (cell_w, lh), (cell_w - lw, lh)],
             "bottom-left": [(0, cell_h - lh), (lw, cell_h - lh), (lw, cell_h), (0, cell_h)],
             "bottom-right": [(cell_w - lw, cell_h - lh), (cell_w, cell_h - lh), (cell_w, cell_h), (cell_w - lw, cell_h)]}
    return [Zone(zone_id=f"label_{c.replace('-', '_')}", camera_id=camera_id, tile_id=tile_id, kind="media", polygon=polys[c])
            for c in spec.label_corners if c in polys]


def label_zone(spec: GridSpec, cell_w: int, cell_h: int, camera_id: str, tile_id: str):
    return label_zones(spec, cell_w, cell_h, camera_id, tile_id)[0]


CANDIDATES = [(2, 2), (3, 3), (4, 4), (1, 2), (2, 1), (2, 3), (3, 2), (2, 4), (4, 2)]


def seam_scores(gray: np.ndarray, rows: int, cols: int, band: int = 3) -> list[float]:
    """For every interior seam of an R×C layout, how much stronger the image gradient is across the
    seam than elsewhere (1.0 = no seam). NVR multiplexes have hard borders between cells."""
    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)       # per column
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)       # per row
    base_x, base_y = float(np.median(dx)) + 1e-3, float(np.median(dy)) + 1e-3
    scores = []
    for c in range(1, cols):
        x = int(round(w * c / cols))
        scores.append(float(dx[max(0, x - band): min(len(dx), x + band)].max()) / base_x)
    for r in range(1, rows):
        y = int(round(h * r / rows))
        scores.append(float(dy[max(0, y - band): min(len(dy), y + band)].max()) / base_y)
    return scores


def detect_grid(frames: list[np.ndarray], min_ratio: float = 7.0) -> tuple[GridSpec | None, dict]:
    """Pick the layout with the MOST cells whose WEAKEST seam is still clearly a border (min seam
    ratio >= min_ratio). A 4×4 contains the 2×2's seams, so it qualifies only if its quarter seams
    are strong too; a 2×2 outranks 1×2 because its horizontal seam also qualifies.
    Returns (spec or None for a single camera, evidence)."""
    grays = [f.mean(axis=2).astype(np.float32) if f.ndim == 3 else f.astype(np.float32) for f in frames]
    evidence = {}
    best = None
    for (r, c) in sorted(CANDIDATES, key=lambda rc: -(rc[0] * rc[1])):
        per_frame = [seam_scores(g, r, c) for g in grays]
        mins = [min(s) for s in per_frame]
        score = float(np.median(mins))
        evidence[f"{r}x{c}"] = round(score, 1)
        if score >= min_ratio and best is None:
            best = (r, c)
    return (GridSpec(rows=best[0], cols=best[1]) if best else None), evidence



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
