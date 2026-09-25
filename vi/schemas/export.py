"""Export JSON Schema for every contract model to ./schemas. Run: python -m vi.schemas.export"""
from __future__ import annotations

import json
from pathlib import Path

from . import (Attributes, ContactSheetResult, EnrichmentPatch, Event, Fact, SceneCard, Tick, Tube)
from .episode import EpisodeClose, EpisodeHeader, EventRecord, PatchRecord, TickRecord, TubeRecord

MODELS = [Tick, Tube, Attributes, EnrichmentPatch, Event, SceneCard, Fact, ContactSheetResult,
          EpisodeHeader, TickRecord, EventRecord, PatchRecord, TubeRecord, EpisodeClose]


def main(out_dir: str | Path = "schemas") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for m in MODELS:
        p = out / f"{m.__name__}.schema.json"
        p.write_text(json.dumps(m.model_json_schema(), indent=2))
        written.append(p)
    print(f"exported {len(written)} schemas to {out}/")
    return written


if __name__ == "__main__":
    main()
