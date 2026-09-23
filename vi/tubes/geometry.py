from __future__ import annotations

import numpy as np

from vi.schemas import Box, FloorPoint

FOOT_UNCERTAINTY_M = 0.2
OCCLUDED_FEET_UNCERTAINTY_M = 1.0


def project_foot(box: Box, H: np.ndarray | None, feet_visible: bool = True,
                 tile_id: str | None = None) -> FloorPoint | None:
    """Ground-plane projection of a tube's foot point. E-TUBE-12: when feet are not
    visible (box truncated at the bottom or behind furniture) the bbox bottom is still
    used but tagged and given a wide uncertainty so fusion loosens its match radius."""
    if H is None:
        return None
    u, v = box.foot_point()
    p = H @ np.array([u, v, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    x, y = float(p[0] / p[2]), float(p[1] / p[2])
    if feet_visible:
        return FloorPoint(x_m=x, y_m=y, tile_id=tile_id, uncertainty_m=FOOT_UNCERTAINTY_M, source="homography")
    return FloorPoint(x_m=x, y_m=y, tile_id=tile_id, uncertainty_m=OCCLUDED_FEET_UNCERTAINTY_M,
                      source="fallback_bbox_bottom")
