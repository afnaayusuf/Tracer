from __future__ import annotations

from vi.schemas import Tube

MIN_LIFE_MS = 1500
MIN_HEIGHT_PX = 48
EDGE_PX = 8


def grade_tube(tube: Tube, frame_w: int, frame_h: int, median_height_px: float | None = None,
               min_life_ms: int = MIN_LIFE_MS, min_height_frac: float = 0.4, border_px: int = EDGE_PX) -> Tube:
    """E-DET-01 / resolution ceiling: a tube that lived under 1.5 s, is under 48 px tall, or was
    born hugging the frame border is real evidence of *something*, not a confirmed person. It
    stays in the store, flagged, so the script can count it apart."""
    if tube.quality_reason and tube.quality_reason.startswith("writer:"):
        return tube                                   # the VLM said this crop is not a person; geometry cannot overrule that
    life = tube.last_seen.corrected_ms() - tube.born.corrected_ms()
    b = tube.box
    reasons = []
    if life < min_life_ms:
        reasons.append(f"life {life / 1000:.1f}s")
    h = max(tube.max_height_px, b.height)
    floor = min_height_frac * median_height_px if median_height_px else MIN_HEIGHT_PX   # scene units when known
    if h < floor:
        reasons.append(f"height {int(h)}px < {int(floor)}px")
    if life < min_life_ms:
        pass
    at_border = b.x1 <= border_px or b.y1 <= border_px or b.x2 >= frame_w - border_px or b.y2 >= frame_h - border_px
    if at_border and life < 2 * min_life_ms:      # a brief flicker at the edge; a long tube that EXITS at the edge is a person
        reasons.append("at frame border")
    tube.quality = "low" if reasons else "ok"
    tube.quality_reason = ", ".join(reasons) or None
    return tube
