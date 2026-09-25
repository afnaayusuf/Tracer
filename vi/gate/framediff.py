from __future__ import annotations

from collections import deque

import numpy as np

from vi.schemas import Box

from .base import GateResult, MotionBlob


class FrameDiffGate:
    """Slice implementation of Ring 0 on decoded grayscale frames. Pure numpy.

    Block grid (default 16 px) mirrors the macroblock grid the MV gate will use, so
    thresholds and downstream ROI packing carry over unchanged when MVGate lands.
    Edge cases handled here: E-GATE-02 (luma step), E-GATE-03 (adaptive noise floor),
    E-GATE-04 (global motion). Stationary blindness (E-GATE-01) is by design and is
    covered by heartbeat detections in Ring 1, not here.
    """

    def __init__(self, camera_id: str, block: int = 16, k_noise: float = 4.0,
                 min_blocks: int = 2, global_fraction: float = 0.6,
                 luma_step_thr: float = 12.0, bg_alpha: float = 0.05, history: int = 30,
                 min_thr: float = 4.0):
        self.camera_id = camera_id
        self.block = block
        self.k_noise = k_noise
        self.min_blocks = min_blocks
        self.global_fraction = global_fraction
        self.luma_step_thr = luma_step_thr
        self.bg_alpha = bg_alpha
        self.min_thr = min_thr
        self._bg: np.ndarray | None = None
        self._prev_luma: float | None = None
        self._noise_hist: deque[float] = deque(maxlen=history)

    def _block_energy(self, diff: np.ndarray) -> np.ndarray:
        h, w = diff.shape
        b = self.block
        hb, wb = h // b, w // b
        d = diff[: hb * b, : wb * b].reshape(hb, b, wb, b)
        return d.mean(axis=(1, 3))

    @staticmethod
    def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        try:  # vectorised path (scipy is present on Colab); the pure-python BFS below is the fallback
            from scipy import ndimage
            labels, n = ndimage.label(mask)
            if n == 0:
                return []
            comps: list[list[tuple[int, int]]] = [[] for _ in range(n)]
            for i, j in zip(*np.nonzero(labels)):
                comps[labels[i, j] - 1].append((int(i), int(j)))
            return comps
        except ImportError:
            pass
        seen = np.zeros_like(mask, dtype=bool)
        comps: list[list[tuple[int, int]]] = []
        hb, wb = mask.shape
        for i in range(hb):
            for j in range(wb):
                if mask[i, j] and not seen[i, j]:
                    stack, comp = [(i, j)], []
                    seen[i, j] = True
                    while stack:
                        ci, cj = stack.pop()
                        comp.append((ci, cj))
                        for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                            if 0 <= ni < hb and 0 <= nj < wb and mask[ni, nj] and not seen[ni, nj]:
                                seen[ni, nj] = True
                                stack.append((ni, nj))
                    comps.append(comp)
        return comps

    def update(self, frame_gray: np.ndarray, t_ms: int) -> GateResult:
        f = frame_gray.astype(np.float32)
        mean_luma = float(f.mean())
        if self._bg is None:
            self._bg = f.copy()
            self._prev_luma = mean_luma
            return GateResult(camera_id=self.camera_id, t_ms=t_ms, blobs=[], mean_luma=mean_luma)

        luma_step = abs(mean_luma - (self._prev_luma or mean_luma)) > self.luma_step_thr
        self._prev_luma = mean_luma
        # E-GATE-02: a global luminance step is a scene-state change; compensate before differencing
        # so a light switching on does not read as whole-frame motion. Median, not mean: a bright
        # object covering a few percent of the frame must not shift the reference.
        diff = np.abs((f - float(np.median(f))) - (self._bg - float(np.median(self._bg))))
        energy = self._block_energy(diff)

        # E-GATE-03: the median block energy of a frame is the sensor/scene noise floor as long as
        # most blocks are static; a rolling median over frames makes it robust to busy frames.
        self._noise_hist.append(float(np.median(energy)))
        noise_floor = float(np.median(self._noise_hist))
        thr = max(noise_floor * self.k_noise, self.min_thr)
        mask = energy > thr
        active_fraction = float(mask.mean())

        global_motion = active_fraction > self.global_fraction and not luma_step
        blobs: list[MotionBlob] = []
        if not global_motion and not luma_step:
            for comp in self._components(mask):
                if len(comp) < self.min_blocks:
                    continue
                ii = [c[0] for c in comp]
                jj = [c[1] for c in comp]
                b = self.block
                box = Box(x1=min(jj) * b, y1=min(ii) * b, x2=(max(jj) + 1) * b, y2=(max(ii) + 1) * b)
                blobs.append(MotionBlob(box=box, energy=float(energy[ii, jj].sum())))

        # background update: slow, and frozen while the frame is globally disturbed
        if not global_motion:
            self._bg = (1 - self.bg_alpha) * self._bg + self.bg_alpha * f
        return GateResult(camera_id=self.camera_id, t_ms=t_ms, blobs=blobs,
                          global_motion=global_motion, luma_step=luma_step,
                          mean_luma=mean_luma, noise_floor=noise_floor,
                          active_fraction=active_fraction)
