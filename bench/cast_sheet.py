"""One image with every tube's birth crop, id, lifetime, state and quality flag, so a human can
say which tubes are the same person or not a person at all. Run after slice_gpu.py.

  python bench/cast_sheet.py data/episodes/*.jsonl --keyframes data/keyframes --out data/bench/cast_sheet.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vi.episode import EpisodeWriter
from vi.schemas.episode import TubeRecord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--keyframes", default="data/keyframes")
    ap.add_argument("--out", default="data/bench/cast_sheet.jpg")
    ap.add_argument("--cell", type=int, default=220)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    tubes = [r.tube for r in EpisodeWriter.read(a.episode) if isinstance(r, TubeRecord) and r.tube.class_label == "person"]
    tubes.sort(key=lambda t: t.born.corrected_ms())
    cols = 5
    rows = (len(tubes) + cols - 1) // cols
    W, H = a.cell, a.cell + 46
    sheet = Image.new("RGB", (cols * W, max(1, rows) * H), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    for i, t in enumerate(tubes):
        x, y = (i % cols) * W, (i // cols) * H
        ref = (t.keyframe_refs or [""])[0]
        path = Path(a.keyframes) / ref.replace("kf://", "") if ref.startswith("kf://") else None
        if path and path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((W - 8, a.cell - 8))
            sheet.paste(im, (x + 4 + (W - 8 - im.width) // 2, y + 4 + (a.cell - 8 - im.height) // 2))
        else:
            dr.text((x + 8, y + a.cell // 2), "no keyframe", fill=(200, 80, 80))
        life = (t.last_seen.corrected_ms() - t.born.corrected_ms()) / 1000
        col = (255, 120, 120) if t.quality == "low" else (180, 255, 180)
        dr.text((x + 6, y + a.cell + 4), f"{t.tube_id}  {t.entity_id or '-'}", fill=(230, 230, 230))
        dr.text((x + 6, y + a.cell + 18), f"born {t.born.corrected_ms() / 1000:.1f}s  life {life:.1f}s  {t.state.value}", fill=(200, 200, 200))
        dr.text((x + 6, y + a.cell + 32), f"{t.quality}{': ' + t.quality_reason if t.quality_reason else ''}", fill=col)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out, quality=88)
    print(f"cast sheet: {len(tubes)} person tubes -> {a.out}")


if __name__ == "__main__":
    main()
