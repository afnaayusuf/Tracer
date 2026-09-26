"""Build-frame profiles (tiers). A profile selects branches of the one engine, the event and zone
vocabularies the compiler and calibration use, sampling rates, and quality thresholds in scene
units. Profiles extend `common`; nothing is per-tier code."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"


class Quality(BaseModel):
    min_life_s: float = 1.5
    min_height_frac: float = Field(0.4, ge=0.0, le=1.0)   # of the camera's median person height
    border_px: int = 8


class Profile(BaseModel):
    name: str
    extends: str | None = None
    tube_classes: list[str] = Field(default_factory=list)
    branches: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    zone_kinds: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    sampling: dict[str, float] = Field(default_factory=lambda: {"active_fps": 4, "quiet_fps": 1})
    quality: Quality = Field(default_factory=Quality)
    notes: str | None = None

    @model_validator(mode="after")
    def _dedupe(self) -> "Profile":
        for f in ("tube_classes", "branches", "events", "zone_kinds", "roles"):
            seen: list[str] = []
            for x in getattr(self, f):
                if x not in seen:
                    seen.append(x)
            setattr(self, f, seen)
        return self


def load_profile(name: str, profiles_dir: Path | str = PROFILES_DIR) -> Profile:
    """Load `<name>.yaml`, merging its parent's lists in front of its own (child adds; child scalars win)."""
    d = Path(profiles_dir)
    raw = yaml.safe_load((d / f"{name}.yaml").read_text())
    child = Profile(**raw)
    if child.extends:
        parent = load_profile(child.extends, d)
        merged = parent.model_dump()
        for f in ("tube_classes", "branches", "events", "zone_kinds", "roles"):
            merged[f] = list(dict.fromkeys(merged[f] + getattr(child, f)))
        merged["sampling"] = {**parent.sampling, **raw.get("sampling", {})}
        merged["quality"] = {**parent.quality.model_dump(), **raw.get("quality", {})}
        merged["name"], merged["extends"], merged["notes"] = child.name, child.extends, child.notes
        return Profile(**merged)
    return child


def available_profiles(profiles_dir: Path | str = PROFILES_DIR) -> list[str]:
    return sorted(p.stem for p in Path(profiles_dir).glob("*.yaml"))
