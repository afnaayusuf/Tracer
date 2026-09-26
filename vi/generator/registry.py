"""The branch registry: every capability the generator can run, what it consumes, at what cadence,
which model backs it, its license, and whether it is measured, planned, or a stub. A profile names
branches; `plan()` resolves them against this registry and the runtime, so a tier can list `pose`
before the pose branch exists and the generator simply reports it as unavailable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Cadence = Literal["frame", "tube_event", "on_demand", "calibration"]
Status = Literal["measured", "planned", "stub"]


@dataclass(frozen=True)
class Branch:
    name: str
    consumes: str                  # "frame" or a tube class ("person", "vehicle", "*")
    cadence: Cadence
    model: str
    license: str
    status: Status
    requires: tuple[str, ...] = field(default_factory=tuple)   # importable modules
    produces: str = ""


BRANCHES: dict[str, Branch] = {b.name: b for b in [
    Branch("detect",   "frame",   "frame",       "RF-DETR nano/medium 1.7.0",          "Apache-2.0", "measured", ("rfdetr",),      "Detection[] per frame"),
    Branch("track",    "frame",   "frame",       "ByteTracker (from source)",           "MIT lineage", "measured", (),              "Tube lifecycle"),
    Branch("relink",   "person",  "tube_event",  "SigLIP 2 base (OSNet-MSMT17 alt.)",  "Apache-2.0", "measured", ("transformers",), "entity ids, relink events"),
    Branch("writer",   "person",  "tube_event",  "Qwen3.5-4B VLM on contact sheets",   "Apache-2.0", "planned",  ("transformers",), "Attributes patch per tube"),
    Branch("pose",     "person",  "tube_event",  "RTMPose (mmpose)",                    "Apache-2.0", "planned",  ("mmpose",),      "keypoints per tube"),
    Branch("actions",  "person",  "tube_event",  "PoseC3D / ST-GCN++ (mmaction2)",      "Apache-2.0", "planned",  ("mmaction",),    "fall, lying, running primitives"),
    Branch("face",     "person",  "on_demand",   "YuNet + SFace (OpenCV Zoo)",          "MIT + Apache-2.0", "planned", ("cv2",),  "face embedding (opt-in)"),
    Branch("ppe",      "person",  "tube_event",  "OWLv2 / Grounding DINO open-vocab",   "Apache-2.0", "planned",  ("transformers",), "helmet/vest/mask flags"),
    Branch("roles",    "person",  "tube_event",  "writer VLM (role field)",             "Apache-2.0", "planned",  ("transformers",), "staff/patient/visitor guess"),
    Branch("density",  "frame",   "on_demand",   "CountGD",                             "MIT",        "planned",  (),               "count in region"),
    Branch("plates",   "vehicle", "tube_event",  "PaddleOCR PP-OCRv6",                  "Apache-2.0", "planned",  ("paddleocr",),   "plate text"),
    Branch("custody",  "*",       "tube_event",  "event compiler (deterministic)",      "-",          "measured", (),               "pickup/drop/custody table"),
]}


def plan(branch_names: list[str], available_modules: set[str] | None = None) -> dict[str, dict]:
    """Resolve profile branches against the registry and (optionally) the runtime. Returns
    name -> {'branch': Branch|None, 'runnable': bool, 'why': str}. Never raises for unknown names:
    a tier may name a capability before it exists; the generator reports it."""
    out: dict[str, dict] = {}
    for n in branch_names:
        b = BRANCHES.get(n)
        if b is None:
            out[n] = {"branch": None, "runnable": False, "why": "not in registry"}
            continue
        if b.status == "stub":
            out[n] = {"branch": b, "runnable": False, "why": "stub"}
            continue
        if available_modules is not None:
            missing = [m for m in b.requires if m not in available_modules]
            if missing:
                out[n] = {"branch": b, "runnable": False, "why": f"missing {missing}"}
                continue
        out[n] = {"branch": b, "runnable": True, "why": b.status}
    return out


def runtime_modules() -> set[str]:
    import importlib.util
    return {m for m in ("rfdetr", "transformers", "mmpose", "mmaction", "cv2", "paddleocr") if importlib.util.find_spec(m)}
