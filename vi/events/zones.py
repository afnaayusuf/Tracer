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


def default_zones(camera_id: str, width: int, height: int, edge_frac: float = 0.1,
                  tile_id: str | None = None) -> list[Zone]:
    """Two exit strips (left/right edges) and a centre floor zone, in native pixels. Enough
    for the first real-footage slice; real zones come from the scene card + walk."""
    e = width * edge_frac
    return [
        Zone(zone_id="exit_left", camera_id=camera_id, tile_id=tile_id, kind="exit",
             polygon=[(0, 0), (e, 0), (e, height), (0, height)]),
        Zone(zone_id="exit_right", camera_id=camera_id, tile_id=tile_id, kind="exit",
             polygon=[(width - e, 0), (width, 0), (width, height), (width - e, height)]),
        Zone(zone_id="floor_centre", camera_id=camera_id, tile_id=tile_id, kind="generic",
             polygon=[(e, height * 0.4), (width - e, height * 0.4), (width - e, height), (e, height)]),
    ]


def load_zones(path: str, camera_id: str | None = None) -> list[Zone]:
    """JSON: [{"zone_id":..., "camera_id":..., "kind":..., "polygon":[[x,y],...], "asset_id":...}, ...]"""
    import json
    from pathlib import Path

    items = json.loads(Path(path).read_text())
    zones = [Zone(**{**z, "polygon": [tuple(p) for p in z["polygon"]]}) for z in items]
    return [z for z in zones if camera_id is None or z.camera_id == camera_id]
