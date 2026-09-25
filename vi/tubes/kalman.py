"""Constant-velocity Kalman filter on a bounding box, written from the equations (no Deep SORT
lineage; see SPEC C2). State x = [cx, cy, a, h, vx, vy, va, vh] with velocities in units per
second, so predict(dt) is correct at any sampled frame rate (E-ING-03 / E-TUBE-13)."""
from __future__ import annotations

import numpy as np

from vi.schemas import Box

STD_POS = 1.0 / 20.0     # position noise as a fraction of box height (SORT convention)
STD_VEL = 1.0 / 160.0    # velocity noise as a fraction of box height per reference frame
REF_FPS = 30.0           # the per-frame noise constants above were tuned at ~30 fps


def box_to_z(b: Box) -> np.ndarray:
    w, h = b.width, b.height
    return np.array([b.x1 + w / 2.0, b.y1 + h / 2.0, w / max(h, 1e-6), h])


def z_to_box(z: np.ndarray) -> Box:
    cx, cy, a, h = float(z[0]), float(z[1]), max(float(z[2]), 1e-3), max(float(z[3]), 1.0)
    w = a * h
    return Box(x1=cx - w / 2.0, y1=cy - h / 2.0, x2=cx + w / 2.0, y2=cy + h / 2.0)


class KalmanBoxFilter:
    def __init__(self, box: Box):
        z = box_to_z(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        std = np.array([2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
                        10 * STD_VEL * h * REF_FPS, 10 * STD_VEL * h * REF_FPS, 1e-5 * REF_FPS,
                        10 * STD_VEL * h * REF_FPS])
        self.P = np.diag(std ** 2)
        self.age_s = 0.0
        self.hits = 1
        self.misses = 0
        self.last_meas_h = z[3]
        self.last_meas_a = z[2]
        self.size_band = (0.6, 1.6)   # E-TUBE-14: predicted h and aspect may not drift outside this band

    def predict(self, dt_s: float) -> Box:
        dt = max(dt_s, 1e-3)
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        h = max(self.x[3], 1.0)
        # process noise: per-frame constants scaled to the elapsed time
        scale = dt * REF_FPS
        std = np.array([STD_POS * h, STD_POS * h, 1e-2, STD_POS * h,
                        STD_VEL * h * REF_FPS, STD_VEL * h * REF_FPS, 1e-5 * REF_FPS, STD_VEL * h * REF_FPS])
        Q = np.diag((std ** 2) * scale)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age_s += dt
        # E-TUBE-14: a box predicted through a long occlusion must not balloon or collapse; a
        # person does not change size while unseen. Clamp size to a band around the last
        # measurement and zero the size velocities once the clamp engages.
        lo, hi = self.size_band
        h_min, h_max = self.last_meas_h * lo, self.last_meas_h * hi
        a_min, a_max = self.last_meas_a * lo, self.last_meas_a * hi
        if not (h_min <= self.x[3] <= h_max):
            self.x[3] = float(np.clip(self.x[3], h_min, h_max)); self.x[7] = 0.0
        if not (a_min <= self.x[2] <= a_max):
            self.x[2] = float(np.clip(self.x[2], a_min, a_max)); self.x[6] = 0.0
        return z_to_box(self.x[:4])

    def update(self, box: Box) -> Box:
        z = box_to_z(box)
        h = max(z[3], 1.0)
        R = np.diag(np.array([STD_POS * h, STD_POS * h, 1e-1, STD_POS * h]) ** 2)
        H = np.zeros((4, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0
        self.last_meas_h = z[3]
        self.last_meas_a = z[2]
        return z_to_box(self.x[:4])

    @property
    def box(self) -> Box:
        return z_to_box(self.x[:4])

    @property
    def speed_px_s(self) -> float:
        return float(np.hypot(self.x[4], self.x[5]))
