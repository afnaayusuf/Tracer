from __future__ import annotations

from pydantic import BaseModel, Field


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


class Zone(BaseModel):
    """Image-space polygon for one camera (world-space zones come with fusion)."""

    zone_id: str
    camera_id: str
    tile_id: str | None = None
    polygon: list[tuple[float, float]] = Field(min_length=3)
    kind: str = "generic"           # generic | exit | asset_home | actuator | media | rest
    asset_id: str | None = None     # for asset_home zones

    def contains(self, pt: tuple[float, float]) -> bool:
        return point_in_polygon(pt, self.polygon)
