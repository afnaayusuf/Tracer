from __future__ import annotations

from vi.schemas import Tube

MIN_LIFE_MS = 1500
MIN_HEIGHT_PX = 48
EDGE_PX = 8


def grade_tube(tube: Tube, frame_w: int, frame_h: int) -> Tube:
    """E-DET-01 / resolution ceiling: a tube that lived under 1.5 s, is under 48 px tall, or was
    born hugging the frame border is real evidence of *something*, not a confirmed person. It
    stays in the store, flagged, so the script can count it apart."""
    life = tube.last_seen.corrected_ms() - tube.born.corrected_ms()
    b = tube.box
    reasons = []
    if life < MIN_LIFE_MS:
        reasons.append(f"life {life / 1000:.1f}s")
    h = max(tube.max_height_px, b.height)
    if h < MIN_HEIGHT_PX:
        reasons.append(f"height {int(h)}px")
    if b.x1 <= EDGE_PX or b.y1 <= EDGE_PX or b.x2 >= frame_w - EDGE_PX or b.y2 >= frame_h - EDGE_PX:
        reasons.append("at frame border")
    tube.quality = "low" if reasons else "ok"
    tube.quality_reason = ", ".join(reasons) or None
    return tube
