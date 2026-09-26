#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 26 build: build-frames (tiers) on the existing engine.
#   * profiles/<tier>.yaml (common + 5 tiers) and vi/profiles.py; vi/generator/registry.py lists every
#     branch with model, license, cadence, status; the slice takes --profile
#   * writer branch (Ring 3b): person crops packed into a numbered contact sheet, one Qwen3.5-4B
#     call per sheet at tube confirmation -> Attributes on the tube -> "looks:" in the scene script
#   * numeric tools in the store (count_entities, entities_present, coverage); the agent must use
#     them for counts/durations; numeric claims are checked and sent back once; coverage % per entity
#   * media zones fire no events; quality thresholds in scene units (fraction of median height)
#   * bench/metamorphic.py: brightness/resolution/fps/flip/time-shift invariants without labels
#   * scenario rule: cited whole-time entities must cover 80% of entities_present(0.9)
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_26.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
PROFILE="${PROFILE:-tier4_industrial}"
SESSION="session 26"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] || warn "GH_TOKEN is not set: will commit locally but cannot push (use the one-cell form)"

step "0. runtime checks"
HAS_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1
if [ "$HAS_GPU" = 1 ]; then echo "GPU: $(nvidia-smi -L | head -1)"; else warn "NO GPU in this runtime -> Runtime > Change runtime type > L4, then rerun"; fi
[ -f "$SOURCE" ] && echo "clip: $SOURCE ($(du -h "$SOURCE" | cut -f1))" || warn "NO CLIP at $SOURCE -> upload HI_DEF_VIDEO.mp4 to /content"

step "1. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin || true
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"
if [ -n "$(git status --porcelain)" ]; then
  warn "uncommitted changes from an interrupted run; discarding them (the script rewrites its files)"
  git checkout -- . && git clean -fdq
fi
git pull -q --ff-only 2>/dev/null || warn "pull skipped (offline or diverged); continuing on local HEAD"
[ -n "$(git log --grep='^session 25' --format=%h)" ] || die "session 25 commit not found; run build_session_25.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/cast_sheet.py
bench/metamorphic.py
bench/reid_eval.py
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/scenario_eval.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
colab/preflight.sh
colab/vllm_venv.sh
edge_cases.yaml
profiles/common.yaml
profiles/tier1_hospital.yaml
profiles/tier2_public.yaml
profiles/tier3_office.yaml
profiles/tier4_industrial.yaml
profiles/tier5_residential.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
tests/scenarios/warehouse.yaml
tests/scenarios/warehouse_identities.yaml
tests/test_detect_roi.py
tests/test_episode.py
tests/test_eval_mot.py
tests/test_events.py
tests/test_fact.py
tests/test_gate.py
tests/test_harness.py
tests/test_ingest.py
tests/test_profiles_generator.py
tests/test_reid.py
tests/test_schemas.py
tests/test_slice_gpu_smoke.py
tests/test_store_agent.py
tests/test_tubes.py
vi/__init__.py
vi/agent/__init__.py
vi/agent/loop.py
vi/agent/tools.py
vi/detect/__init__.py
vi/detect/base.py
vi/detect/fake.py
vi/detect/rfdetr.py
vi/detect/roi.py
vi/episode/__init__.py
vi/episode/debug.py
vi/episode/keyframes.py
vi/episode/writer.py
vi/eval/__init__.py
vi/eval/mot.py
vi/events/__init__.py
vi/events/compiler.py
vi/events/zones.py
vi/gate/__init__.py
vi/gate/base.py
vi/gate/framediff.py
vi/gate/heartbeat.py
vi/generator/__init__.py
vi/generator/registry.py
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
vi/profiles.py
vi/reid/__init__.py
vi/reid/base.py
vi/schemas/__init__.py
vi/schemas/common.py
vi/schemas/contact_sheet.py
vi/schemas/episode.py
vi/schemas/event.py
vi/schemas/export.py
vi/schemas/fact.py
vi/schemas/scene_card.py
vi/schemas/tick.py
vi/schemas/tube.py
vi/store/__init__.py
vi/store/db.py
vi/store/loader.py
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/linker.py
vi/tubes/quality.py
vi/tubes/simple_iou.py
vi/writer/__init__.py
vi/writer/contact_sheet.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p profiles vi/generator vi/writer tests/scenarios colab/sessions data/bench
cat > profiles/common.yaml << 'EOF_VI'
# Branches every tier runs. A tier profile extends this; the generator only runs branches that are
# both listed here-or-in-the-tier and marked available in the registry for this runtime.
name: common
tube_classes: [person, dog, cat, bicycle, car, motorcycle, bus, truck, backpack, handbag, suitcase, umbrella]
branches:
  - detect            # RF-DETR nano, per sampled frame
  - track             # ByteTracker, per frame
  - relink            # SigLIP appearance, tube events
  - writer            # contact-sheet VLM: clothing, carried items, role; tube events
events: [enter_zone, exit_zone, dwell, approach, meet, pickup, drop, left_behind, asset_missing_from_home,
         handoff, impossible_transition, illumination_change, door_state_change, modality_switch,
         signal_lost, signal_restored, camera_moved_suspect, relink]
zone_kinds: [generic, exit, asset_home, media, actuator]
sampling: {active_fps: 4, quiet_fps: 1}
quality:                     # scene units, not pixels (E-DET-01): fractions of this camera's median person height
  min_life_s: 1.5
  min_height_frac: 0.4
  border_px: 8
EOF_VI
cat > profiles/tier1_hospital.yaml << 'EOF_VI'
name: tier1_hospital
extends: common
tube_classes: [person, wheelchair, bed, chair]
branches: [pose, actions, ppe, roles]
events: [fall, lying, wandering, crowd, loiter]
zone_kinds: [bed, corridor, dispenser, triage, exit]
roles: [staff, patient, visitor]
notes: |
  Gait metrics (cadence, step length, symmetry) are rules over RTMPose keypoints and are flags, not
  clinical findings, until validated on that site. Face branch off unless the site opts in.
EOF_VI
cat > profiles/tier2_public.yaml << 'EOF_VI'
name: tier2_public
extends: common
branches: [pose, actions, density, plates]
events: [crowd, loiter, run, left_behind, queue]
zone_kinds: [entrance, exit, queue, shopfront, parking]
notes: |
  Fire/smoke has no commercial-licensed open detector worth shipping; the writer VLM on keyframes at
  low cadence is the honest option and is what `inspect` does.
EOF_VI
cat > profiles/tier3_office.yaml << 'EOF_VI'
name: tier3_office
extends: common
branches: [roles]
events: [tailgate, after_hours, occupancy]
zone_kinds: [door, desk, meeting_room, exit]
EOF_VI
cat > profiles/tier4_industrial.yaml << 'EOF_VI'
name: tier4_industrial
extends: common
tube_classes: [person, forklift, truck, car, suitcase, backpack]
branches: [pose, actions, ppe]
events: [fall, run, proximity, restricted_zone]
zone_kinds: [packing_table, conveyor, dock_aisle, dock_doors, coat_rack, restricted, media, exit]
notes: The warehouse clip is a tier-4 scene; data/zones/HI_DEF_VIDEO.json uses these zone kinds.
EOF_VI
cat > profiles/tier5_residential.yaml << 'EOF_VI'
name: tier5_residential
extends: common
tube_classes: [person, dog, cat, backpack, handbag, suitcase, bicycle]
branches: [roles, custody]
events: [pickup, drop, left_behind, asset_missing_from_home, doorbell, fall]
zone_kinds: [doorstep, hallway, kitchen, entry_shelf, exit, media]
EOF_VI
cat > vi/profiles.py << 'EOF_VI'
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
EOF_VI
cat > vi/generator/__init__.py << 'EOF_VI'
from .registry import BRANCHES, Branch, plan, runtime_modules
EOF_VI
cat > vi/generator/registry.py << 'EOF_VI'
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
EOF_VI
cat > vi/writer/__init__.py << 'EOF_VI'
from .contact_sheet import WRITER_PROMPT, WriterVLM, pack_sheet, parse_sheet_reply
EOF_VI
cat > vi/writer/contact_sheet.py << 'EOF_VI'
"""Ring 3b, the writer branch: pack person crops into one numbered grid, ask the VLM once, get one
JSON record per cell under the Attributes schema (R22). Runs at tube events, never per frame.
The VLM never sees a whole frame here; it sees crops with cell numbers, so it cannot confuse
people across cells without saying which cell it means."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from vi.schemas import Attributes, Modality
from vi.schemas.contact_sheet import CellResult, ContactSheetResult

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "multicolor"]

WRITER_PROMPT = (
    "This image is a grid of numbered cells; each cell shows one person cropped from a camera. "
    "For EVERY cell, describe that person only. Reply with a JSON array, one object per cell, no prose:\n"
    '[{"cell_id": 0, "top_color": one of ' + str(COLORS) + ' or null, "bottom_color": same or null, '
    '"headwear": short text or null, "carried_item": short text or null, "role": short text or null, '
    '"description": at most 12 words, "confidence": 0-1}]\n'
    "Use null when a field is not visible. Cells are numbered top-left to bottom-right starting at 0."
)


def pack_sheet(crops: list[np.ndarray], cell: int = 224, cols: int = 4):
    """Grid with hard borders and a number in each cell (E-FOV-04)."""
    from PIL import Image, ImageDraw
    n = len(crops)
    cols = min(cols, max(1, n))
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(crops):
        x, y = (i % cols) * cell, (i // cols) * cell
        im = Image.fromarray(np.ascontiguousarray(c)).convert("RGB")
        im.thumbnail((cell - 12, cell - 28))
        sheet.paste(im, (x + 6 + (cell - 12 - im.width) // 2, y + 22 + (cell - 28 - im.height) // 2))
        dr.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=(255, 255, 255), width=2)
        dr.rectangle([x + 3, y + 3, x + 40, y + 20], fill=(255, 215, 0))
        dr.text((x + 8, y + 5), f"#{i}", fill=(0, 0, 0))
    return sheet


def _color(v):
    return v.lower() if isinstance(v, str) and v.lower() in COLORS else None


def parse_sheet_reply(text: str, tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
    """Tolerant parse of the VLM reply into the schema; None if it cannot be made valid (E-FOV-07)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end < 0:
        return None
    try:
        items = json.loads(t[start:end + 1])
    except Exception:
        return None
    cells: list[CellResult] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            cid = int(it.get("cell_id"))
        except Exception:
            continue
        if not (0 <= cid < len(tube_ids)):
            continue
        desc_bits = [str(it.get("description") or "")[:120]]
        if it.get("headwear"):
            desc_bits.append(f"headwear: {str(it['headwear'])[:40]}")
        if it.get("role"):
            desc_bits.append(f"role: {str(it['role'])[:30]}")
        conf = it.get("confidence", 0.6)
        conf = float(conf) if isinstance(conf, (int, float)) else 0.6
        kwargs = dict(modality=modality, carried_item=(str(it["carried_item"])[:60] if it.get("carried_item") else None),
                      carried_item_confidence=0.6 if it.get("carried_item") else 0.0,
                      description="; ".join(b for b in desc_bits if b)[:240], confidence=max(0.0, min(1.0, conf)))
        if modality.has_color:
            kwargs.update(top_color=_color(it.get("top_color")), bottom_color=_color(it.get("bottom_color")))
        else:
            kwargs.update(color_reason="ir_mode")
        try:
            cells.append(CellResult(cell_id=cid, tube_id=tube_ids[cid], attributes=Attributes(**kwargs)))
        except Exception:
            continue
    # fill missing cells with an empty attributes record so the sheet validates (per-cell failure, not sheet failure)
    have = {c.cell_id for c in cells}
    for i, tid in enumerate(tube_ids):
        if i not in have:
            cells.append(CellResult(cell_id=i, tube_id=tid, attributes=Attributes(
                modality=modality, confidence=0.0, **({} if modality.has_color else {"color_reason": "ir_mode"}))))
    cells.sort(key=lambda c: c.cell_id)
    try:
        return ContactSheetResult(sheet_id=f"sheet_{int(time.time() * 1000)}", modality=modality,
                                  expected_cells=len(tube_ids), cells=cells)
    except Exception:
        return None


class WriterVLM:
    """Qwen3.5 (or any transformers image-text model) on a packed sheet. `backend` may be an
    existing TransformersBackend to reuse its loaded model and processor."""

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", backend=None, max_new_tokens: int = 700, device: str | None = None):
        import torch
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        if backend is not None and getattr(backend, "model", None) is not None:
            self.model, self.proc, self.device = backend.model, backend.tok, backend.device
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(model_id)
        self.calls = 0
        self.last_ms = 0.0

    def describe(self, crops: list[np.ndarray], tube_ids: list[str], modality: Modality = Modality.rgb) -> ContactSheetResult | None:
        sheet = pack_sheet(crops)
        messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": WRITER_PROMPT}]}]
        t0 = time.perf_counter()
        try:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        text = tok.decode(gen, skip_special_tokens=True)
        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return parse_sheet_reply(text, tube_ids, modality)
EOF_VI
cat > vi/events/compiler.py << 'EOF_VI'
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from vi.gate.base import GateResult
from vi.schemas import CamTime, Event, EventType, TubeSnapshot

from .zones import Zone

PERSON_CLASSES = {"person"}


def _eid(*parts: object) -> str:
    return "ev_" + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class _Membership:
    inside_run: int = 0
    outside_run: int = 0
    is_inside: bool = False
    entered_at_ms: int | None = None
    dwell_fired: bool = False


@dataclass
class _AssetState:
    present: bool | None = None
    last_present_ms: int | None = None
    visitors: dict[str, int] = field(default_factory=dict)   # tube_id -> last ms seen inside


class EventCompiler:
    """Ring 3a. Deterministic predicates over tube snapshots, zones and gate results.
    No model calls. One instance per camera.

    E-EVT-01 hysteresis: enter after `enter_ticks` consecutive inside ticks, exit after
             `exit_ticks` consecutive outside ticks.
    E-EVT-03 pickup: fires only on a heartbeat that (a) reports the asset absent, (b) is
             taken while no person is inside the asset-home zone, and (c) follows a
             heartbeat that reported it present. The subject is whoever was inside the
             zone in between; if nobody was, the event degrades to
             asset_missing_from_home with no subject rather than blaming someone.
    E-GATE-02 luma step -> illumination_change (scene-state), never object motion.
    E-KB-01 sustained global motion -> camera_moved_suspect.
    """

    def __init__(self, camera_id: str, zones: list[Zone], enter_ticks: int = 2, exit_ticks: int = 2,
                 dwell_ms: int = 10_000, tile_id: str | None = None, offset_ms: int = 0,
                 moved_after_updates: int = 10, open_grace_ms: int = 1500):
        self.camera_id = camera_id
        self.tile_id = tile_id
        self.zones = {z.zone_id: z for z in zones if z.camera_id == camera_id}
        self.enter_ticks = enter_ticks
        self.exit_ticks = exit_ticks
        self.dwell_ms = dwell_ms
        self.offset_ms = offset_ms
        self.moved_after_updates = moved_after_updates
        self._mem: dict[tuple[str, str], _Membership] = {}
        self._assets: dict[str, _AssetState] = {
            z.zone_id: _AssetState() for z in self.zones.values() if z.kind == "asset_home"}
        self._global_run = 0
        self._moved_fired = False
        self._ticks_seen = 0
        self.open_grace_ms = open_grace_ms
        self._open_t_ms: int | None = None

    def _t(self, t_ms: int) -> CamTime:
        return CamTime(cam_utc_ms=t_ms, offset_ms=self.offset_ms)

    def _event(self, type_: EventType, t_ms: int, zone_id: str | None, subjects: list[str],
               objects: list[str] | None = None, payload: dict | None = None,
               confidence: float = 1.0) -> Event:
        t_bucket = t_ms // 1000
        return Event(event_id=_eid(self.camera_id, type_.value, zone_id, ",".join(sorted(subjects)), t_bucket),
                     type=type_, t=self._t(t_ms), camera_id=self.camera_id, tile_id=self.tile_id,
                     zone_id=zone_id, subject_tube_ids=subjects, object_ids=objects or [],
                     payload=payload or {}, confidence=confidence,
                     dedupe_key=f"{type_.value}|{zone_id}|{t_bucket}")

    # ---------------- tube predicates ----------------
    def on_tick(self, tubes: list[TubeSnapshot], t_ms: int) -> list[Event]:
        events: list[Event] = []
        live = {s.tube_id for s in tubes}
        if self._open_t_ms is None:
            self._open_t_ms = t_ms
        self._ticks_seen += 1
        if t_ms - self._open_t_ms <= self.open_grace_ms:
            # E-EVT-10: whoever is already in frame when the episode opens did not "enter";
            # seed zone memberships silently (for a short grace window, since the gate needs
            # a frame or two to warm up) so enter_zone means a boundary was crossed.
            for snap in tubes:
                foot = snap.box.foot_point()
                for z in self.zones.values():
                    if z.contains(foot):
                        m = self._mem.setdefault((snap.tube_id, z.zone_id), _Membership())
                        m.is_inside, m.inside_run, m.entered_at_ms = True, self.enter_ticks, t_ms
                        if z.kind == "asset_home" and snap.class_label in PERSON_CLASSES:
                            self._assets[z.zone_id].visitors[snap.tube_id] = t_ms
            return events
        for snap in tubes:
            foot = snap.box.foot_point()
            for z in self.zones.values():
                if z.kind == "media":
                    continue                    # E-DET-05: props and screens are not places anyone enters
                key = (snap.tube_id, z.zone_id)
                m = self._mem.setdefault(key, _Membership())
                inside = z.contains(foot)
                if inside:
                    m.inside_run += 1
                    m.outside_run = 0
                    if z.kind == "asset_home" and snap.class_label in PERSON_CLASSES:
                        self._assets[z.zone_id].visitors[snap.tube_id] = t_ms
                    if not m.is_inside and m.inside_run >= self.enter_ticks:
                        m.is_inside = True
                        m.entered_at_ms = t_ms
                        m.dwell_fired = False
                        events.append(self._event(EventType.enter_zone, t_ms, z.zone_id, [snap.tube_id]))
                    elif m.is_inside and not m.dwell_fired and m.entered_at_ms is not None \
                            and t_ms - m.entered_at_ms >= self.dwell_ms:
                        m.dwell_fired = True
                        events.append(self._event(EventType.dwell, t_ms, z.zone_id, [snap.tube_id],
                                                  payload={"dwell_ms": t_ms - m.entered_at_ms}))
                else:
                    m.outside_run += 1
                    m.inside_run = 0
                    if m.is_inside and m.outside_run >= self.exit_ticks:
                        m.is_inside = False
                        events.append(self._event(EventType.exit_zone, t_ms, z.zone_id, [snap.tube_id],
                                                  payload={"inside_ms": t_ms - (m.entered_at_ms or t_ms)}))
        # tubes that vanished while inside a zone: close their membership silently (exit is a
        # tube-lifecycle matter, handled by fusion/lost state, not a zone exit)
        for key in [k for k in self._mem if k[0] not in live]:
            del self._mem[key]
        return events

    # ---------------- asset heartbeat ----------------
    def on_heartbeat(self, zone_id: str, asset_present: bool, t_ms: int,
                     persons_inside: list[str]) -> list[Event]:
        z = self.zones[zone_id]
        st = self._assets[zone_id]
        events: list[Event] = []
        if asset_present:
            st.present = True
            st.last_present_ms = t_ms
            st.visitors.clear()
            return events
        # absent
        if persons_inside:
            return events  # E-EVT-03(b): someone may be occluding the shelf; wait
        if st.present is True:
            since = st.last_present_ms or 0
            suspects = [tid for tid, last in st.visitors.items() if last >= since]
            if suspects:
                events.append(self._event(EventType.pickup, t_ms, zone_id, suspects, objects=[z.asset_id or zone_id],
                                          payload={"window_ms": [since, t_ms]}, confidence=0.8 if len(suspects) == 1 else 0.5))
            else:
                events.append(self._event(EventType.asset_missing_from_home, t_ms, zone_id, [],
                                          objects=[z.asset_id or zone_id], payload={"window_ms": [since, t_ms]},
                                          confidence=0.9))
        st.present = False
        return events

    # ---------------- ingest ----------------
    def on_size_change(self, t_ms: int, old: tuple[int, int] | None, new: tuple[int, int]) -> list[Event]:
        """E-ING-05: resolution change => homography and zones invalid; same flag as a moved camera."""
        return [self._event(EventType.camera_moved_suspect, t_ms, None, [],
                            payload={"reason": "resolution_change", "old": list(old or ()), "new": list(new)},
                            confidence=0.95)]

    # ---------------- gate / scene state ----------------
    def on_gate(self, g: GateResult) -> list[Event]:
        events: list[Event] = []
        if g.luma_step:
            events.append(self._event(EventType.illumination_change, g.t_ms, None, [],
                                      payload={"mean_luma": g.mean_luma}))
        if g.global_motion:
            self._global_run += 1
            if self._global_run >= self.moved_after_updates and not self._moved_fired:
                self._moved_fired = True
                events.append(self._event(EventType.camera_moved_suspect, g.t_ms, None, [],
                                          payload={"consecutive_global_updates": self._global_run}, confidence=0.7))
        else:
            self._global_run = 0
        return events
EOF_VI
cat > vi/tubes/quality.py << 'EOF_VI'
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
    if b.x1 <= border_px or b.y1 <= border_px or b.x2 >= frame_w - border_px or b.y2 >= frame_h - border_px:
        reasons.append("at frame border")
    tube.quality = "low" if reasons else "ok"
    tube.quality_reason = ", ".join(reasons) or None
    return tube
EOF_VI
cat > vi/agent/tools.py << 'EOF_VI'
"""Deterministic Block 2 tools (R26): the reasoning model calls these; it never sees video.
search_* are SQL filters; get_script renders the scene script the model reads; clip returns
evidence references. Every result carries the ids the model must cite (R27)."""
from __future__ import annotations

import time

from sqlalchemy import and_, or_, select
from sqlalchemy.engine import Engine

from vi.store.db import entities, episodes, events, insert_ignore, scripts, tubes


def _ts(ms: int, t0: int = 0) -> str:
    s = max(0, ms - t0) / 1000.0
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def search_events(engine: Engine, *, tile_id: str | None = None, camera_id: str | None = None,
                  t_start_ms: int | None = None, t_end_ms: int | None = None, event_type: str | list[str] | None = None,
                  zone_id: str | None = None, entity_id: str | None = None, tube_id: str | None = None,
                  limit: int = 200) -> list[dict]:
    q = select(events)
    conds = []
    if tile_id: conds.append(events.c.tile_id == tile_id)
    if camera_id: conds.append(events.c.camera_id == camera_id)
    if t_start_ms is not None: conds.append(events.c.t_ms >= t_start_ms)
    if t_end_ms is not None: conds.append(events.c.t_ms <= t_end_ms)
    if event_type:
        types = [event_type] if isinstance(event_type, str) else list(event_type)
        conds.append(events.c.type.in_(types))
    if zone_id: conds.append(events.c.zone_id == zone_id)
    if conds:
        q = q.where(and_(*conds))
    q = q.order_by(events.c.t_ms).limit(limit * 4 if (entity_id or tube_id) else limit)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(q)]
    if tube_id:
        rows = [r for r in rows if tube_id in (r["subject_tube_ids"] or [])]
    if entity_id:
        tube_ids = set(_tubes_of_entity(engine, entity_id))
        rows = [r for r in rows if entity_id in (r["subject_entity_ids"] or []) or tube_ids & set(r["subject_tube_ids"] or [])]
    return rows[:limit]


def search_tubes(engine: Engine, *, tile_id: str | None = None, camera_id: str | None = None,
                 class_label: str | None = None, t_start_ms: int | None = None, t_end_ms: int | None = None,
                 entity_id: str | None = None, named: str | None = None, min_life_ms: int = 0, limit: int = 200) -> list[dict]:
    q = select(tubes)
    conds = []
    if tile_id: conds.append(tubes.c.tile_id == tile_id)
    if camera_id: conds.append(tubes.c.camera_id == camera_id)
    if class_label: conds.append(tubes.c.class_label == class_label)
    if entity_id: conds.append(tubes.c.entity_id == entity_id)
    if named: conds.append(tubes.c.named == named)
    if t_start_ms is not None: conds.append(tubes.c.last_seen_ms >= t_start_ms)   # overlaps the window
    if t_end_ms is not None: conds.append(tubes.c.born_ms <= t_end_ms)
    if min_life_ms: conds.append(tubes.c.last_seen_ms - tubes.c.born_ms >= min_life_ms)
    if conds:
        q = q.where(and_(*conds))
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q.order_by(tubes.c.born_ms).limit(limit))]


def search_entities(engine: Engine, *, camera_id: str | None = None, class_label: str | None = None,
                    t_start_ms: int | None = None, t_end_ms: int | None = None, named: str | None = None,
                    limit: int = 200) -> list[dict]:
    q = select(entities)
    conds = []
    if camera_id: conds.append(entities.c.camera_id == camera_id)
    if class_label: conds.append(entities.c.class_label == class_label)
    if named: conds.append(entities.c.named == named)
    if t_start_ms is not None: conds.append(entities.c.last_seen_ms >= t_start_ms)
    if t_end_ms is not None: conds.append(entities.c.first_seen_ms <= t_end_ms)
    if conds:
        q = q.where(and_(*conds))
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q.order_by(entities.c.first_seen_ms).limit(limit))]


def episode_window(engine: Engine, episode_id: str) -> tuple[int, int]:
    with engine.connect() as conn:
        row = conn.execute(select(episodes.c.t0_ms, episodes.c.t1_ms).where(episodes.c.episode_id == episode_id)).first()
    if row is None:
        raise KeyError(episode_id)
    return int(row[0]), int(row[1] if row[1] is not None else row[0])


def coverage(engine: Engine, episode_id: str, class_label: str = "person", confirmed_only: bool = True) -> list[dict]:
    """Per entity: seen interval, coverage of the episode window (0-1), tube count, quality.
    Deterministic; this is what the agent must use for 'how long', 'whole time', 'how many'."""
    t0, t1 = episode_window(engine, episode_id)
    span = max(1, t1 - t0)
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(and_(tubes.c.episode_id == episode_id, tubes.c.class_label == class_label)))]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["entity_id"], []).append(r)
    out = []
    for eid, rs in by.items():
        ok = any(r.get("quality", "ok") == "ok" for r in rs)
        if confirmed_only and not ok:
            continue
        # union of tube intervals (clipped to the window)
        ivs = sorted((max(t0, r["born_ms"]), min(t1, r["last_seen_ms"])) for r in rs)
        covered, cur = 0, None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: covered += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: covered += cur[1] - cur[0]
        out.append({"entity_id": eid, "first_seen_ms": min(r["born_ms"] for r in rs), "last_seen_ms": max(r["last_seen_ms"] for r in rs),
                    "coverage": round(covered / span, 3), "tubes": len(rs), "quality": "ok" if ok else "low"})
    return sorted(out, key=lambda x: (-x["coverage"], x["first_seen_ms"]))


def count_entities(engine: Engine, episode_id: str, t_start_ms: int | None = None, t_end_ms: int | None = None,
                   class_label: str = "person", confirmed_only: bool = True) -> dict:
    """Distinct confirmed entities present in [t_start, t_end] (default: the whole episode)."""
    t0, t1 = episode_window(engine, episode_id)
    a, b = (t0 if t_start_ms is None else t_start_ms), (t1 if t_end_ms is None else t_end_ms)
    rows = coverage(engine, episode_id, class_label, confirmed_only)
    present = [r["entity_id"] for r in rows if r["last_seen_ms"] >= a and r["first_seen_ms"] <= b]
    return {"count": len(present), "entity_ids": present, "window_ms": [a, b], "confirmed_only": confirmed_only}


def entities_present(engine: Engine, episode_id: str, min_coverage: float = 0.9, class_label: str = "person") -> dict:
    """Entities seen for at least min_coverage of the episode: the deterministic answer to
    'who stayed the whole time' (0.9 tolerates short occlusions)."""
    rows = coverage(engine, episode_id, class_label)
    ids = [r["entity_id"] for r in rows if r["coverage"] >= min_coverage]
    return {"min_coverage": min_coverage, "count": len(ids), "entity_ids": ids,
            "coverage": {r["entity_id"]: r["coverage"] for r in rows}}


def _tubes_of_entity(engine: Engine, entity_id: str) -> list[str]:
    with engine.connect() as conn:
        row = conn.execute(select(entities.c.tube_ids).where(entities.c.entity_id == entity_id)).first()
    return list(row[0]) if row else []


def clip(engine: Engine, *, entity_id: str | None = None, tube_id: str | None = None, mode: str = "crop") -> dict:
    """Evidence references for an entity or tube: keyframe crops now; video segments when the
    retention tier exists (E-STO-03). Returns refs, never pixels."""
    with engine.connect() as conn:
        if tube_id:
            rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.tube_id == tube_id))]
        elif entity_id:
            rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.entity_id == entity_id).order_by(tubes.c.born_ms))]
        else:
            raise ValueError("entity_id or tube_id required")
    refs = [k for r in rows for k in (r["keyframe_refs"] or [])]
    return {"mode": mode, "entity_id": entity_id or (rows[0]["entity_id"] if rows else None),
            "tube_ids": [r["tube_id"] for r in rows], "keyframe_refs": refs,
            "segments": [{"camera_id": r["camera_id"], "t_start_ms": r["born_ms"], "t_end_ms": r["last_seen_ms"]} for r in rows],
            "available": bool(refs), "note": None if refs else "no keyframes stored for this subject"}


def get_script(engine: Engine, episode_id: str, max_events: int = 400, cache: bool = True) -> str:
    """The scene script: what the reasoning model reads instead of video. Deterministic, compact,
    every line carries the ids it can cite. Cached in the scripts table."""
    with engine.connect() as conn:
        if cache:
            cached = conn.execute(select(scripts.c.text).where(scripts.c.episode_id == episode_id)).first()
            if cached:
                return cached[0]
        ep = conn.execute(select(episodes).where(episodes.c.episode_id == episode_id)).first()
        if ep is None:
            raise KeyError(episode_id)
        ep = dict(ep._mapping)
        tube_rows = [dict(r._mapping) for r in conn.execute(select(tubes).where(tubes.c.episode_id == episode_id).order_by(tubes.c.born_ms))]
        ev_rows = [dict(r._mapping) for r in conn.execute(select(events).where(events.c.episode_id == episode_id).order_by(events.c.t_ms).limit(max_events))]
    t0 = ep["t0_ms"]
    tube_ent = {r["tube_id"]: r["entity_id"] for r in tube_rows}
    by_entity: dict[str, list[dict]] = {}
    for r in tube_rows:
        by_entity.setdefault(r["entity_id"], []).append(r)
    def ent_line(eid: str, rows: list[dict]) -> str:
        first, last = min(r["born_ms"] for r in rows), max(r["last_seen_ms"] for r in rows)
        name = rows[0]["named"] or ("anonymous" if eid.startswith("anon:") else "unnamed")
        kf = next((k for r in rows for k in (r["keyframe_refs"] or [])), None)
        attrs = next((r["attributes"] for r in rows if r.get("attributes")), None)
        desc = ""
        if attrs:
            bits = [attrs.get("description") or "", f"top {attrs['top_color']}" if attrs.get("top_color") else "",
                    f"carrying {attrs['carried_item']}" if attrs.get("carried_item") else ""]
            desc = "  looks: " + "; ".join(b for b in bits if b)
        return (f"  {eid}  {rows[0]['class_label']}  {name}  tubes {','.join(r['tube_id'] for r in rows)}  "
                f"seen {_ts(first, t0)}–{_ts(last, t0)}  coverage {cov(rows):.0%}  states {','.join(sorted({r['state'] for r in rows}))}"
                + (f"  keyframe {kf}" if kf else "") + desc)

    span = max(1, (ep["t1_ms"] or t0) - t0)
    def cov(rows: list[dict]) -> float:
        ivs = sorted((max(t0, r["born_ms"]), min(ep["t1_ms"] or t0, r["last_seen_ms"])) for r in rows)
        total, cur = 0, None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: total += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur: total += cur[1] - cur[0]
        return total / span
    confirmed = {e: rows for e, rows in by_entity.items() if any(r.get("quality", "ok") == "ok" for r in rows)}
    brief = {e: rows for e, rows in by_entity.items() if e not in confirmed}
    people = sum(1 for rows in confirmed.values() if rows[0]["class_label"] == "person")
    lines = [f"EPISODE {ep['episode_id']} | tile {ep['tile_id']} | cameras {','.join(ep['camera_ids'] or [])} | "
             f"{_ts(t0, t0)}–{_ts(ep['t1_ms'] or t0, t0)} | status {ep['status']} | kb v{ep['kb_version']}",
             f"CAST: {len(confirmed)} confirmed entities ({people} people), {len(brief)} brief sightings, {len(tube_rows)} tubes"]
    for eid, rows in confirmed.items():
        lines.append(ent_line(eid, rows))
    if brief:
        lines.append("BRIEF SIGHTINGS (low quality: too short, too small or at the frame border; not counted as people):")
        for eid, rows in brief.items():
            reason = next((r.get("quality_reason") for r in rows if r.get("quality_reason")), "")
            lines.append(ent_line(eid, rows) + (f"  why: {reason}" if reason else ""))
    lines.append(f"TIMELINE: {len(ev_rows)} events")
    for e in ev_rows:
        subj = e["subject_tube_ids"] or []
        who = ", ".join(f"{tube_ent.get(t, 'anon:' + t)} (tube {t})" for t in subj) or "—"
        extra = ""
        if e["type"] == "relink" and e["payload"]:
            extra = f" sim {e['payload'].get('similarity')}"
        elif e["type"] == "dwell" and e["payload"]:
            extra = f" {e['payload'].get('dwell_ms', 0) / 1000:.1f}s"
        elif e["type"] in ("pickup", "asset_missing_from_home", "drop") and e["object_ids"]:
            extra = f" object {','.join(e['object_ids'])}"
        lines.append(f"  {_ts(e['t_ms'], t0)}  {e['type']:<22s} zone {e['zone_id'] or '-':<14s} {who}{extra}  [{e['event_id']}]")
    text = "\n".join(lines)
    if cache:
        with engine.begin() as conn:
            insert_ignore(conn, scripts, [dict(episode_id=episode_id, text=text, rendered_at_ms=int(time.time() * 1000))])
    return text
EOF_VI
cat > vi/agent/__init__.py << 'EOF_VI'
from .loop import FakeBackend, OpenAIBackend, TransformersBackend, ask, extract_json
from .tools import (clip, count_entities, coverage, entities_present, get_script, search_entities, search_events,
                    search_tubes)
EOF_VI
cat > vi/agent/loop.py << 'EOF_VI'
"""Block 2 agent loop (R26/R27): a reasoning model that never sees video, only tool results, and
must cite ids for every claim. Every model turn is one JSON `AgentStep` under a schema, so it is
grammar-constrainable (R3) on any backend that supports structured outputs.

Backends implement `complete(messages, schema) -> str`. `FakeBackend` is a deterministic
scripted planner for tests and CPU runs; `OpenAIBackend` talks to any OpenAI-compatible server
(vLLM serve, SGLang) and asks for JSON-schema structured output.
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy.engine import Engine

from vi.schemas import EventType

from . import tools as T

EVENT_TYPES = [e.value for e in EventType]

TOOL_SPECS = {
    "search_events": "Events by time window / zone / type / entity. args: tile_id, camera_id, t_start_ms, t_end_ms, "
                     "event_type (one of EVENT_TYPES, or a list), zone_id, entity_id, tube_id, limit",
    "search_tubes": "Tubes (per-camera tracks). args: tile_id, camera_id, class_label, t_start_ms, t_end_ms, entity_id, named, min_life_ms, limit",
    "search_entities": "Entities (people/objects across tubes). args: camera_id, class_label, t_start_ms, t_end_ms, named, limit",
    "get_script": "The scene script of an episode (cast + timeline). args: episode_id",
    "clip": "Evidence references (keyframes, time segments) for an entity or tube. args: entity_id or tube_id",
    "count_entities": "Distinct confirmed people in a window. args: t_start_ms, t_end_ms (omit for whole episode)",
    "entities_present": "Entities seen for at least min_coverage of the episode ('stayed the whole time'). args: min_coverage (default 0.9)",
    "coverage": "Per-entity seen interval and coverage fraction. args: none",
}


class ToolStep(BaseModel):
    action: Literal["tool"] = "tool"
    tool: Literal["search_events", "search_tubes", "search_entities", "get_script", "clip", "count_entities", "entities_present", "coverage"]
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = Field("", max_length=400)


ANSWER_MAX_CHARS = 4000


class AnswerStep(BaseModel):
    action: Literal["answer"] = "answer"
    text: str = Field(max_length=ANSWER_MAX_CHARS)
    citations: list[str] = Field(default_factory=list, description="entity ids, tube ids or event ids from tool results")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ClarifyStep(BaseModel):
    action: Literal["clarify"] = "clarify"
    question: str = Field(max_length=300)


AgentStep = Annotated[Union[ToolStep, AnswerStep, ClarifyStep], Field(discriminator="action")]
step_adapter: TypeAdapter = TypeAdapter(AgentStep)
STEP_SCHEMA = step_adapter.json_schema()

SYSTEM = """You answer questions about video by calling tools over an evidence store; you never see video.
Rules: (1) the episode's scene script is given to you; call tools only when it does not answer the question;
(2) every claim in an answer must cite ids
(entity ids like cam1:E3, tube ids like cam1:4000:10, event ids like ev_...) that appeared in tool results;
(3) if nothing matches, say so and suggest how to widen the search; never invent people, times or events;
(4) if the question is ambiguous (which person, which time), ask one clarifying question;
(5) answer over entities, not tubes; a person may have several tubes; (6) times are episode-relative
mm:ss.s in scripts and absolute milliseconds in tool args; (7) any count, duration or "whole time" claim MUST come from
count_entities / entities_present / coverage, never from reading the script. Respond with exactly one JSON object per turn:
{"action":"tool","tool":...,"args":{...},"why":...} | {"action":"answer","text":...,"citations":[...],"confidence":...}
| {"action":"clarify","question":...}. Count people from CONFIRMED entities; BRIEF SIGHTINGS are not people.
Keep answers under 60 words and cite at most 8 ids; cite entity ids (cam1:E7), not tube ids, unless asked about tubes; mention only events
that appear in the script or tool results.
EVENT_TYPES: """ + ", ".join(EVENT_TYPES) + "\nTools: " + json.dumps(TOOL_SPECS)

ID_RE = re.compile(r"\b(ev_[0-9a-f]{16}|[A-Za-z0-9_]+:E\d+|anon:[A-Za-z0-9_:]+|[A-Za-z0-9_]+:\d+:\d+)\b")


class FakeBackend:
    """Scripted planner: script -> one search chosen from the question -> cited answer. Enough to
    exercise the loop, the citation check and the empty-result path without a model."""

    name = "fake"

    def __init__(self, bogus_citation: bool = False):
        self.bogus = bogus_citation

    def complete(self, messages: list[dict], schema: dict) -> str:
        user = [m for m in messages if m["role"] == "user"]
        q = user[0]["content"].split("Question:")[-1].lower()     # the question, not the inlined script
        n_tool_results = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("TOOL RESULT"))
        has_script = "SCENE SCRIPT:" in user[0]["content"]
        if n_tool_results == 0 and not has_script:
            ep = re.search(r"episode (ep_[0-9a-f]+)", user[0]["content"], re.IGNORECASE)
            return json.dumps({"action": "tool", "tool": "get_script", "args": {"episode_id": ep.group(1) if ep else ""}, "why": "read the script"})
        if n_tool_results == (0 if has_script else 1):
            if "key" in q or "pickup" in q or "took" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "pickup"}, "why": "custody"})
            if "nobody" in q or "unicorn" in q:
                return json.dumps({"action": "tool", "tool": "search_events", "args": {"event_type": "fall"}, "why": "probe"})
            return json.dumps({"action": "tool", "tool": "search_entities", "args": {"class_label": "person"}, "why": "who"})
        last = messages[-1]["content"]
        ids = ID_RE.findall(last)
        if not ids:
            return json.dumps({"action": "answer", "text": "No matching events in this episode; widen the time window or check another tile.",
                               "citations": [], "confidence": 0.9})
        cites = ["ev_deadbeefdeadbeef"] if self.bogus else sorted(set(ids))[:4]
        return json.dumps({"action": "answer", "text": f"Found {len(set(ids))} matching record(s); see citations.",
                           "citations": cites, "confidence": 0.8})


class OpenAIBackend:
    """Any OpenAI-compatible chat server (vLLM serve, SGLang). Requests JSON-schema structured
    output; falls back to vLLM's guided_json extra field on servers that predate response_format."""

    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "Qwen/Qwen3.5-4B",
                 api_key: str = "EMPTY", temperature: float = 0.0, max_tokens: int = 600, timeout: float = 120.0):
        import urllib.request
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self._req = urllib.request

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = self._req.Request(self.base_url + "/chat/completions", data=data, method="POST",
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with self._req.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def complete(self, messages: list[dict], schema: dict) -> str:
        base = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}      # Qwen3.x: no reasoning preamble before the JSON
        try:
            out = self._post({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}})
        except Exception:
            try:
                out = self._post({**base, "guided_json": schema})
            except Exception:
                base.pop("chat_template_kwargs", None)                   # servers that reject the field
                out = self._post({**base, "guided_json": schema})
        return extract_json(out["choices"][0]["message"]["content"])


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> str:
    """Take the first balanced {...} object out of a model reply (think blocks, code fences,
    prose, and trailing text are all tolerated). Returns the raw text if none is found, so
    validation fails loudly rather than silently."""
    t = THINK_RE.sub("", text).strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start = t.find("{")
    if start < 0:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return text


class TransformersBackend:
    """In-process fallback when no server can be started: loads the model with transformers in the
    runtime's own torch. No structured-output grammar; the loop validates and retries instead."""

    name = "transformers"

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 300, device: str | None = None, warmup: bool = True):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        last = None
        self.model = None
        try:
            import fla  # noqa: F401  (flash-linear-attention: fused kernels for Qwen3.5's GDN layers)
            self.fused = True
        except Exception:
            self.fused = False
        for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                cls = getattr(__import__("transformers", fromlist=[loader]), loader)
                try:
                    self.model = cls.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
                except Exception:
                    self.model = cls.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
                self.loader = loader
                break
            except Exception as e:  # pragma: no cover
                last = e
        self.last_gen_tokens = 0
        self.last_tok_s = 0.0
        if self.model is None:
            raise RuntimeError(f"could not load {model_id}: {last!r}")
        try:
            self.tok = AutoProcessor.from_pretrained(model_id)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_id)
        if warmup:   # CUDA context, kernels and cache allocation happen here, not inside the first question
            try:
                self.complete([{"role": "system", "content": "Reply with {}"}, {"role": "user", "content": "{}"}], {})
            except Exception:
                pass

    def complete(self, messages: list[dict], schema: dict) -> str:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\nReply with the JSON object only, no prose."}
        try:   # Qwen3.x templates: thinking is on by default and eats the whole token budget
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        dt = max(1e-6, time.perf_counter() - t0)
        self.last_gen_tokens = int(getattr(gen, "shape", [len(gen)])[0])
        self.last_tok_s = self.last_gen_tokens / dt
        tokenizer = getattr(self.tok, "tokenizer", self.tok)
        return extract_json(tokenizer.decode(gen, skip_special_tokens=True))


def _flatten_ids(x: Any) -> list[str]:
    """citations may arrive as strings, ints, dicts ({"entity_id": ...}) or nested lists"""
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return ID_RE.findall(str(x)) or ([str(x)] if isinstance(x, str) else [])
    if isinstance(x, dict):
        return [i for v in x.values() for i in _flatten_ids(v)]
    if isinstance(x, list):
        return [i for v in x for i in _flatten_ids(v)]
    return []


EVENT_WORDS = {"pickup": ["pickup", "pick-up", "picked up", "picking up", "picks up"], "drop": ["drop event", "dropped"],
               "fall": ["fall event", "fell", "falling"], "left_behind": ["left behind"], "loiter": ["loiter"],
               "crowd": ["crowd event"], "run": ["running event"], "handoff": ["handoff", "hand-off"],
               "impossible_transition": ["impossible transition"], "relink": ["relink"]}


def contradicted_event_claims(engine: Engine, episode_id: str, text: str) -> list[str]:
    """E-AGT-06 for prose: an answer that talks about an event type the episode does not contain
    is asserting evidence that does not exist. Returns the offending event types."""
    # sentence-level, ignoring negated mentions ("nobody fell", "no pickup events") which assert absence
    neg = re.compile(r"\b(no|not|nobody|none|never|didn't|did not|wasn't|weren't|isn't|aren't|without)\b")
    named: set[str] = set()
    for sent in re.split(r"(?<=[.!?;])\s+", text.lower()):
        if neg.search(sent):
            continue
        for et, words in EVENT_WORDS.items():
            if any(w in sent for w in words):
                named.add(et)
    if not named:
        return []
    have = {e["type"] for e in T.search_events(engine, limit=10000) if e["episode_id"] == episode_id}
    return sorted(et for et in named if et not in have)


WHOLE_TIME_RE = re.compile(r"whole (time|episode|clip)|entire (time|episode|clip)|throughout", re.I)
COUNT_Q_RE = re.compile(r"how many|number of (people|persons|workers)|count", re.I)
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}


def _numbers_in(text: str) -> list[int]:
    t = ID_RE.sub(" ", text)
    t = re.sub(r"\d{2}:\d{2}(\.\d)?", " ", t)                        # timestamps are not counts
    nums = [int(m) for m in re.findall(r"(?<![\d.:])(\d{1,3})(?![\d.:])", t)]
    nums += [v for w, v in NUM_WORDS.items() if re.search(rf"\b{w}\b", t, re.I)]
    return nums


def numeric_claim_issue(engine: Engine, episode_id: str, question: str, step: AnswerStep) -> dict | None:
    """E-AGT-06 for numbers: a people count or a 'stayed the whole time' list in the answer is
    compared with the deterministic tools; a disagreement is sent back once with the true values."""
    try:
        if COUNT_Q_RE.search(question):
            truth = T.count_entities(engine, episode_id)["count"]
            nums = _numbers_in(step.text)
            if nums and truth not in nums and all(abs(n - truth) > 1 for n in nums[:3]):
                return {"msg": f"count_entities says {truth} confirmed people in this episode (your answer implies {nums[:3]})."}
        if WHOLE_TIME_RE.search(question):
            ep = T.entities_present(engine, episode_id, 0.9)
            truth_ids = set(ep["entity_ids"])
            cited = {c for c in step.citations if re.search(r":E\d+$", c)}
            if truth_ids and len(cited & truth_ids) < max(1, len(truth_ids) - 1):
                return {"msg": f"entities_present(min_coverage=0.9) says {len(truth_ids)} entities stayed the whole time: "
                               f"{sorted(truth_ids)}; your answer names {sorted(cited & truth_ids) or 'none of them'}."}
    except Exception:
        return None
    return None


def _salvage(raw: str) -> AnswerStep | ClarifyStep | None:
    """Accept an answer that only failed on length or a missing optional field; never a tool
    call, which must be exact. Retrying seven times to trim a paragraph is not a behaviour."""
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("action") == "answer" and isinstance(obj.get("text"), str):
        cites = _flatten_ids(obj.get("citations")) + ID_RE.findall(obj["text"])   # objects, nested lists, ids in prose
        seen: list[str] = []
        for c in cites:
            if c not in seen:
                seen.append(c)
        return AnswerStep(text=obj["text"][:ANSWER_MAX_CHARS], citations=seen[:50],
                          confidence=float(obj.get("confidence", 0.5)) if isinstance(obj.get("confidence"), (int, float)) else 0.5)
    if obj.get("action") == "clarify" and isinstance(obj.get("question"), str):
        return ClarifyStep(question=obj["question"][:300])
    return None


EPISODE_TOOLS = {"count_entities", "entities_present", "coverage"}


def run_tool(engine: Engine, step: ToolStep, episode_id: str | None = None) -> Any:
    fn = {"search_events": T.search_events, "search_tubes": T.search_tubes, "search_entities": T.search_entities,
          "get_script": T.get_script, "clip": T.clip, "count_entities": T.count_entities,
          "entities_present": T.entities_present, "coverage": T.coverage}[step.tool]
    import inspect
    allowed = set(inspect.signature(fn).parameters) - {"engine", "episode_id"} if step.tool != "get_script" else {"episode_id"}
    dropped = [k for k in step.args if k not in allowed]
    args = {k: v for k, v in step.args.items() if v is not None and k in allowed}
    if step.tool == "search_events" and args.get("event_type"):
        types = args["event_type"] if isinstance(args["event_type"], list) else [args["event_type"]]
        bad = [t for t in types if t not in EVENT_TYPES]
        if bad:
            raise ValueError(f"unknown event_type {bad}; valid: {', '.join(EVENT_TYPES)}")
    if step.tool == "get_script":
        return fn(engine, args.get("episode_id", ""))
    if step.tool in EPISODE_TOOLS:
        result = fn(engine, episode_id or step.args.get("episode_id", ""), **args)
        if dropped:
            result = {**result, "note": f"ignored unknown args {dropped}"}
        return result
    result = fn(engine, **args)
    if dropped and isinstance(result, list):
        result = [{"note": f"ignored unknown args {dropped}"}] + result if result else [{"note": f"ignored unknown args {dropped}; no matches"}]
    return result


def _compact(result: Any, limit: int = 6000) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def ask(engine: Engine, question: str, episode_id: str, backend, max_steps: int = 6,
        inline_script: bool = True) -> dict:
    """Run the loop. With inline_script the scene script is placed in the first user message so
    the model's first turn is already a search or an answer (one round trip saved) and the
    server's prefix cache holds system+script across questions on the same episode. Returns the
    final step, the trace, the citation audit (E-AGT-06) and latency."""
    t_start = time.perf_counter()
    seen_ids: set[str] = set()
    first = f"Episode {episode_id}."
    if inline_script:
        script = T.get_script(engine, episode_id)
        seen_ids.update(ID_RE.findall(script))
        first += f"\n\nSCENE SCRIPT:\n{script}"
    first += f"\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    trace: list[dict] = []
    final: dict | None = None
    model_ms, tool_ms = 0.0, 0.0
    for i in range(max_steps):
        t0 = time.perf_counter()
        raw = backend.complete(messages, STEP_SCHEMA)
        model_ms += (time.perf_counter() - t0) * 1000
        try:
            step = step_adapter.validate_json(raw)
        except ValidationError as e:
            step = _salvage(raw)                    # an over-long or slightly malformed answer is still an answer
            if step is None:
                reason = e.errors()[0]["msg"]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"TOOL RESULT: invalid step ({reason}); reply with one valid JSON object."})
                trace.append({"step": i, "invalid": raw[:200], "reason": reason})
                continue
            trace.append({"step": i, "salvaged": e.errors()[0]["msg"]})
        messages.append({"role": "assistant", "content": raw})
        if isinstance(step, ToolStep):
            result: Any = None
            t0 = time.perf_counter()
            try:
                result = run_tool(engine, step, episode_id)
                text = _compact(result)
                err = None
            except Exception as e:
                text, err = f"error: {type(e).__name__}: {e}", str(e)
            tool_ms += (time.perf_counter() - t0) * 1000
            seen_ids.update(ID_RE.findall(text))
            n = len(result) if isinstance(result, list) else (1 if result else 0)
            trace.append({"step": i, "tool": step.tool, "args": step.args, "results": n, "error": err})
            messages.append({"role": "user", "content": f"TOOL RESULT ({step.tool}, {n} item(s)):\n{text}"})
            continue
        if isinstance(step, ClarifyStep):
            final = {"action": "clarify", "question": step.question}
            break
        valid = [c for c in step.citations if c in seen_ids]
        invalid = [c for c in step.citations if c not in seen_ids]
        already_revised = any("revise" in t for t in trace)
        bad_claims = contradicted_event_claims(engine, episode_id, step.text)
        num_issue = numeric_claim_issue(engine, episode_id, question, step)
        if num_issue and not already_revised:
            trace.append({"step": i, "revise": "numeric claim disagrees with tools", "detail": num_issue["msg"]})
            messages.append({"role": "user", "content": "TOOL RESULT: " + num_issue["msg"] + " Answer again using these numbers."})
            continue
        if bad_claims and not already_revised:
            trace.append({"step": i, "revise": "event claim without evidence", "claims": bad_claims})
            messages.append({"role": "user", "content": f"TOOL RESULT: your answer mentions {bad_claims} but this episode contains no "
                                                        f"events of that type. Remove or correct that claim; cite only events that exist."})
            continue
        if invalid and not valid and not already_revised:
            trace.append({"step": i, "revise": "citations not in tool results", "invalid": invalid})
            messages.append({"role": "user", "content": "TOOL RESULT: your citations do not appear in any tool result. "
                                                        "Answer again citing only ids you were shown, or say nothing matched."})
            continue
        final = {"action": "answer", "text": step.text, "citations": valid, "rejected_citations": invalid,
                 "confidence": step.confidence, "cited": bool(valid), "unsupported_event_claims": bad_claims,
                 "numeric_issue": num_issue["msg"] if num_issue else None}
        break
    if final is None:
        final = {"action": "answer", "text": "I could not complete this within the step budget.", "citations": [], "cited": False,
                 "confidence": 0.0}
    total_ms = (time.perf_counter() - t_start) * 1000
    return {"question": question, "episode_id": episode_id, "backend": getattr(backend, "name", "?"),
            "steps": len(trace), "trace": trace, "final": final,
            "latency": {"total_ms": round(total_ms), "model_ms": round(model_ms), "tool_ms": round(tool_ms),
                        "turns": sum(1 for t in trace if "tool" in t or "revise" in t or "invalid" in t) + 1,
                        "first_prompt_chars": len(first),
                        "gen_tokens": getattr(backend, "last_gen_tokens", None), "tok_s": round(getattr(backend, "last_tok_s", 0.0), 1) or None,
                        "fused_kernels": getattr(backend, "fused", None)}}
EOF_VI
cat > bench/slice_gpu.py << 'EOF_VI'
"""Real-footage slice (Day 4): reader -> gate -> ROIs -> RF-DETR -> tubes -> events -> episode file,
with a native-res keyframe saved at every tube birth. One row to data/bench/slice_gpu.jsonl.

  python bench/slice_gpu.py --source clip.mp4 --detect hybrid      # ROIs + 1 Hz full-frame heartbeat (default)
  python bench/slice_gpu.py --source clip.mp4 --detect roi         # motion ROIs only (session 04/05 behaviour)
  python bench/slice_gpu.py --source clip.mp4 --detect frame       # full frame every tick (upper bound on recall)
  python bench/slice_gpu.py --source clip.mp4 --zones data/zones/cam1.json --tile lobby
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vi.detect import (blobs_to_rois, crop_roi, dedupe_detections, full_frame_roi, is_tube_class, pad_batch,
                       remap_detections)
try:
    from vi.detect.rfdetr import RFDETRDetector
except ImportError:  # CPU runtime: only --model fake works
    RFDETRDetector = None  # type: ignore
from vi.episode import EpisodeWriter, KeyframeStore, annotate
from vi.events import EventCompiler, default_zones, load_zones
from vi.gate import FrameDiffGate, HeartbeatScheduler
from vi.ingest import VideoReader
from vi.schemas import CamTime, Provenance, Tick, TubeSnapshot
from vi.schemas.episode import CastMember, EpisodeStatus
from vi.reid import HistogramEmbedder, crop_for_embedding, make_embedder
from vi.profiles import load_profile
from vi.tubes import TRACKERS, TubeLinker, grade_tube


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--tile", default="tile1")
    ap.add_argument("--model", default="nano")
    ap.add_argument("--fps", type=float, default=4.0, help="R11 range 2-5; IoU-only tracking collapses below ~4")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="detector floor; ByteTracker re-attaches 0.1-0.5 detections and births only >=0.5")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="byte")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from timing (JIT profiling runs)")
    ap.add_argument("--detect", choices=["roi", "frame", "hybrid"], default="frame",
                    help="frame (default, measured best at one camera): full frame every tick; roi: motion ROIs only; "
                         "hybrid: ROIs + full-frame heartbeat (multi-camera cost lever, under investigation)")
    ap.add_argument("--heartbeat-ms", type=int, default=1000, help="hybrid: full-frame detection period on active tiles")
    ap.add_argument("--quiet-heartbeat-ms", type=int, default=10000)
    ap.add_argument("--debug-frames", type=int, default=0, help="save N annotated frames to data/debug/<episode>/")
    ap.add_argument("--reid", choices=["none", "auto", "hist", "siglip", "osnet"], default="none",
                    help="appearance embeddings + TubeLinker (E-TUBE-04); auto = osnet > siglip > hist")
    ap.add_argument("--reid-every-ticks", type=int, default=8, help="gallery refresh cadence for active tubes")
    ap.add_argument("--reid-sim", type=float, default=0.88, help="from bench/reid_eval.py (SigLIP on the warehouse clip)")
    ap.add_argument("--reid-near-sim", type=float, default=0.85)
    ap.add_argument("--profile", default="common", help="build-frame profile (profiles/<name>.yaml): classes, branches, quality")
    ap.add_argument("--writer", choices=["none", "qwen", "fake"], default="none", help="contact-sheet attributes at tube confirmation")
    ap.add_argument("--writer-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--writer-batch", type=int, default=8, help="crops per sheet")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default: data/zones/<clip stem>.json if present, else edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--skip-s", type=float, default=0.0, help="ignore the first N seconds (metamorphic time shift)")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
    profile = load_profile(a.profile)
    tube_classes = set(profile.tube_classes)
    print(f"profile: {profile.name} branches={profile.branches}")
    reader = VideoReader(a.camera, a.source, target_fps=a.fps, max_width=640, want_rgb=True)
    batch = 1 if a.detect == "frame" else a.batch          # frame mode: one image per tick, trace at 1
    if a.model == "fake":
        from vi.detect.fake import BrightBlobDetector
        det = BrightBlobDetector(threshold=a.threshold, batch_size=batch)   # CPU smoke path (tests, no GPU)
    else:
        det = RFDETRDetector(size=a.model, threshold=a.threshold, batch_size=batch)
    gate = FrameDiffGate(a.camera)
    kf = KeyframeStore(Path(a.out).parent / "keyframes")
    current = {"frame": None}
    zones = None
    tracker = compiler = writer = ep = None
    stage = Counter()
    detect_ms: list[float] = []
    ev_types: Counter = Counter()
    births = 0
    closed_all = []
    tick_ms = int(1000 / a.fps)
    frames = 0
    hb = HeartbeatScheduler(active_ms=a.heartbeat_ms, quiet_ms=a.quiet_heartbeat_ms)
    heartbeats = 0
    births_by_origin: Counter = Counter()
    confirmed_by_origin: Counter = Counter()
    rebirths = 0                                   # tube born next to one that just died = fragmentation, counted directly
    recent_dead: list[tuple[int, float, float, float]] = []   # (t_ms, cx, cy, h)
    seen_ids: set[str] = set()
    confirmed_ids: set[str] = set()
    origin_of: dict[str, str] = {}
    debug_every = max(1, int(a.fps * 1.5)) if a.debug_frames else 0     # one frame every ~1.5 s until N saved
    duplicate_pairs: list[dict] = []                # two live person tubes on one body: the hybrid bug, caught in the act
    dup_frames: list[str] = []
    vlm_writer = None
    if a.writer == "qwen":
        try:
            from vi.writer import WriterVLM
            vlm_writer = WriterVLM(model_id=a.writer_model)
            print(f"[writer] {a.writer_model} loaded")
        except Exception as e:
            print(f"[writer] unavailable ({type(e).__name__}: {str(e)[:100]}); continuing without attributes")
    elif a.writer == "fake":
        from vi.writer.contact_sheet import parse_sheet_reply
        class _Fake:
            calls = 0; last_ms = 0.0
            def describe(self, crops, tube_ids, modality=None):
                self.calls += 1
                import json
                return parse_sheet_reply(json.dumps([{"cell_id": i, "top_color": "orange", "description": "worker in a vest", "confidence": 0.7}
                                                     for i in range(len(tube_ids))]), tube_ids)
        vlm_writer = _Fake()
    described: set[str] = set()
    writer_ms: list[float] = []
    embedder = make_embedder(a.reid) if a.reid != "none" else None
    aux_embedder = HistogramEmbedder() if embedder and embedder.name != "hist" else None
    linker = TubeLinker(a.camera, sim_thr=a.reid_sim, near_sim_thr=a.reid_near_sim) if embedder else None
    absorbed_total = 0
    embed_ms: list[float] = []
    last_embed_tick: dict[str, int] = {}
    pending_link: dict[str, np.ndarray] = {}
    pending_aux: dict[str, np.ndarray] = {}
    relink_events = 0
    debug_paths: list[str] = []
    person_dets: list[int] = []
    concurrent_persons: list[int] = []
    state_ticks: Counter = Counter()

    for fr in reader.frames():
        if frames >= a.max_frames:
            break
        if a.skip_s and fr.pts_ms < a.skip_s * 1000:
            continue
        h, w = fr.rgb.shape[:2]
        current["frame"] = fr.rgb
        if zones is None:   # first frame: zones need the native size
            zones_path = a.zones or (str(Path("data/zones") / (Path(a.source).stem + ".json")) if (Path("data/zones") / (Path(a.source).stem + ".json")).exists() else None)
            zones = load_zones(zones_path, a.camera) if zones_path else default_zones(a.camera, w, h, tile_id=a.tile)
            media_zones = [z for z in zones if z.kind == "media"]
            print(f"zones: {[z.zone_id for z in zones]} ({'file ' + zones_path if zones_path else 'defaults'})")
            exit_boxes = [z.polygon for z in zones if z.kind == "exit"]
            from vi.schemas import Box
            exits = [Box(x1=min(p[0] for p in poly), y1=min(p[1] for p in poly),
                         x2=max(p[0] for p in poly), y2=max(p[1] for p in poly)) for poly in exit_boxes]
            tkw = {"iou_thr": a.iou_thr} if a.tracker == "simple" else {}
            tracker = TRACKERS[a.tracker](a.camera, max_occluded_ms=a.max_occluded_ms, exit_boxes=exits,
                                          keyframe_sink=kf.make_sink(lambda: current["frame"]), **tkw)
            compiler = EventCompiler(a.camera, zones, enter_ticks=1, exit_ticks=2, dwell_ms=5000, tile_id=a.tile)
            writer = EpisodeWriter(a.out)
            ep = writer.open(a.tile, [a.camera], CamTime(cam_utc_ms=fr.pts_ms), prov)
        t0 = time.perf_counter()
        g = gate.update(fr.gray, fr.pts_ms)
        stage["gate"] += time.perf_counter() - t0
        events = compiler.on_gate(g)
        t0 = time.perf_counter()
        det_source = "detector"
        rois = blobs_to_rois(g.blobs, w, h, stride=fr.stride) if a.detect in ("roi", "hybrid") else []
        tile_active = len(tracker._tracks) > 0
        if a.detect == "frame" or (a.detect == "hybrid" and hb.due(fr.pts_ms, tile_active)):
            rois.append(full_frame_roi(w, h))                   # heartbeat = one more crop in the batch
            det_source = "heartbeat"
            heartbeats += 1
        dets = []
        for i in range(0, len(rois), batch):
            chunk = rois[i:i + batch]
            crops, real = pad_batch([crop_roi(fr.rgb, r) for r in chunk], batch)
            for r, d in zip(chunk, det.detect_batch(crops)[:real]):
                dets += remap_detections(d, r, w, h)
        dets = dedupe_detections([d for d in dets if d.class_label in tube_classes])  # E-DET-10, classes from the profile
        if media_zones:   # E-DET-05: jackets on a rack, posters, screens are not people
            dets = [d for d in dets if not any(z.contains(d.box.foot_point()) for z in media_zones)]
        person_dets.append(sum(1 for d in dets if d.class_label == "person" and d.confidence >= 0.5))
        dt_detect = time.perf_counter() - t0
        if frames >= a.warmup:
            stage["detect"] += dt_detect
            detect_ms.append(dt_detect * 1000)
        t0 = time.perf_counter()
        before = len(tracker._tracks)
        live, closed = tracker.update(dets, fr.pts_ms, det_source=det_source)
        # provenance + rebirth accounting (person tubes only)
        for t in live:
            if t.class_label != "person":
                continue
            if t.tube_id not in seen_ids:
                seen_ids.add(t.tube_id)
                origin = next((d.origin for d in dets if d.box.iou(t.box) > 0.7), "unknown")
                births_by_origin[origin] += 1
                origin_of[t.tube_id] = origin
                cx, cy = (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2
                if any(fr.pts_ms - tm <= 1500 and ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 <= hh for tm, x, y, hh in recent_dead):
                    rebirths += 1
            if t.state.value == "active" and t.tube_id not in confirmed_ids:
                confirmed_ids.add(t.tube_id)
                confirmed_by_origin[origin_of.get(t.tube_id, "unknown")] += 1
        if linker is not None:
            # embed at first sight, but only *link* once the tracker has confirmed the tube (state
            # active): an unconfirmed tube can still be deleted next tick, and a link to a tube
            # that never existed corrupts the cast (seen in session 12).
            due_birth = [t for t in live if t.class_label == "person" and t.tube_id not in last_embed_tick]
            due_refresh = [t for t in live if t.class_label == "person" and t.state.value == "active"
                           and t.tube_id in last_embed_tick and t.tube_id not in pending_link
                           and frames - last_embed_tick[t.tube_id] >= a.reid_every_ticks]
            todo = due_birth + due_refresh
            if todo:
                t0 = time.perf_counter()
                crops = [crop_for_embedding(fr.rgb, t.box) for t in todo]
                embs = embedder.embed(crops)
                auxs = aux_embedder.embed(crops) if aux_embedder else [None] * len(todo)
                embed_ms.append((time.perf_counter() - t0) * 1000)
                birth_ids = {x.tube_id for x in due_birth}
                for t, e, ax in zip(todo, embs, auxs):
                    last_embed_tick[t.tube_id] = frames
                    if t.tube_id in birth_ids:
                        pending_link[t.tube_id] = e
                        if ax is not None:
                            pending_aux[t.tube_id] = ax
                    else:
                        linker.on_refresh(t, e, ax)
            live_ids = {t.tube_id for t in live}
            for t in live:
                if t.class_label == "person" and t.tube_id not in pending_link:
                    linker.on_state(t, fr.pts_ms)          # occluded tubes become relink candidates
            for t in live:
                if t.tube_id in pending_link and t.state.value == "active":
                    ev = linker.on_birth(t, pending_link.pop(t.tube_id), fr.pts_ms, pending_aux.pop(t.tube_id, None))
                    if ev is not None:
                        events.append(ev)
                        relink_events += 1
                        print(f"t={fr.pts_ms:7d}  relink                 {ev.subject_tube_ids[0]} -> {ev.subject_tube_ids[1]} sim={ev.payload['similarity']} thr={ev.payload['threshold']}")
            for tid in [k for k in pending_link if k not in live_ids]:
                pending_link.pop(tid); pending_aux.pop(tid, None)    # deleted while unconfirmed: never linked
            for ghost in linker.absorbed:                             # the occluded tube a link replaced
                dead = tracker.drop(ghost)
                if dead is not None:
                    closed_all.append(dead)
                    absorbed_total += 1
            linker.absorbed.clear()
            live = [t for t in live if t.tube_id in tracker._tracks]
            for t in live:
                if t.class_label == "person":
                    t.entity_id = linker.entity_of(t.tube_id)
            for t in closed:
                mev = linker.on_close(t, fr.pts_ms)
                if mev is not None:
                    events.append(mev)
                    print(f"t={fr.pts_ms:7d}  merge                  {mev.payload['merged_entity']} -> {mev.subject_entity_ids[0]} sim={mev.payload['similarity']}")
        if vlm_writer is not None and frames % 4 == 0:  # tube-event cadence: newly confirmed people, at most every 4 ticks
            todo = [t for t in live if t.class_label == "person" and t.state.value == "active" and t.tube_id not in described][: a.writer_batch]
            if todo:
                res = vlm_writer.describe([crop_for_embedding(fr.rgb, t.box, pad=0.15) for t in todo], [t.tube_id for t in todo])
                writer_ms.append(getattr(vlm_writer, "last_ms", 0.0))
                if res is not None:
                    by_tube = {c.tube_id: c.attributes for c in res.cells}
                    for t in todo:
                        if t.tube_id in by_tube and by_tube[t.tube_id].confidence > 0:
                            t.attributes = by_tube[t.tube_id]
                            print(f"t={fr.pts_ms:7d}  describe               {t.tube_id}: {t.attributes.description}"
                                  + (f" | top {t.attributes.top_color.value}" if t.attributes.top_color else ""))
                for t in todo:
                    described.add(t.tube_id)
        persons = [t for t in live if t.class_label == "person" and t.state.value in ("active", "born")]
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                ti, tj = persons[i], persons[j]
                iou = ti.box.iou(tj.box)
                if iou >= 0.5 and len(duplicate_pairs) < 12:
                    dets_here = [{"origin": d.origin, "conf": round(d.confidence, 2), "box": [round(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                                  "iou_i": round(d.box.iou(ti.box), 2), "iou_j": round(d.box.iou(tj.box), 2)}
                                 for d in dets if d.class_label == "person" and (d.box.iou(ti.box) > 0.1 or d.box.iou(tj.box) > 0.1)]
                    duplicate_pairs.append({"t_ms": fr.pts_ms, "hb": det_source == "heartbeat", "iou": round(iou, 2),
                                            "a": {"id": ti.tube_id, "origin": origin_of.get(ti.tube_id), "state": ti.state.value,
                                                  "box": [round(v) for v in (ti.box.x1, ti.box.y1, ti.box.x2, ti.box.y2)]},
                                            "b": {"id": tj.tube_id, "origin": origin_of.get(tj.tube_id), "state": tj.state.value,
                                                  "box": [round(v) for v in (tj.box.x1, tj.box.y1, tj.box.x2, tj.box.y2)]},
                                            "dets_on_tick": dets_here})
                    if len(dup_frames) < 4:
                        p = annotate(fr.rgb, rois, dets, live, f"DUPLICATE t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                                     Path(a.out).parent / "debug" / ep / f"dup_{frames:04d}.jpg")
                        if p:
                            dup_frames.append(str(p))
        for t in closed:
            if t.class_label == "person":
                recent_dead.append((fr.pts_ms, (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2, t.box.height))
        recent_dead = [r for r in recent_dead if fr.pts_ms - r[0] <= 1500]
        if debug_every and frames % debug_every == 0 and len(debug_paths) < a.debug_frames:
            p = annotate(fr.rgb, rois, dets, live, f"t={fr.pts_ms}ms {a.detect} hb={'Y' if det_source == 'heartbeat' else 'n'}",
                         Path(a.out).parent / "debug" / ep / f"tick_{frames:04d}.jpg")
            if p:
                debug_paths.append(str(p))
        concurrent_persons.append(sum(1 for t in live if t.class_label == "person" and t.state.value == "active"))
        state_ticks.update(t.state.value for t in live if t.class_label == "person")
        births += max(0, len(live) + len(closed) - before) if closed else max(0, len(live) - before)
        closed_all += closed
        stage["track"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        snaps = [TubeSnapshot(tube_id=t.tube_id, class_label=t.class_label, state=t.state, box=t.box,
                              det_source="detector" if t.state.value == "active" else "predicted") for t in live]
        events += compiler.on_tick(snaps, fr.pts_ms)
        tick = Tick(camera_id=a.camera, tile_id=a.tile, tick_index=frames, t_start=CamTime(cam_utc_ms=fr.pts_ms),
                    t_end=CamTime(cam_utc_ms=fr.pts_ms + tick_ms), tubes=snaps,
                    event_ids=[e.event_id for e in events], gate_energy=sum(b.energy for b in g.blobs), provenance=prov)
        writer.write_tick(ep, tick)
        for e in events:
            writer.write_event(ep, e)
            ev_types[e.type.value] += 1
            print(f"t={fr.pts_ms:7d}  {e.type.value:22s} zone={e.zone_id} subjects={e.subject_tube_ids}")
        stage["compile"] += time.perf_counter() - t0
        frames += 1

    if frames == 0:
        raise SystemExit("no frames read; check --source")
    tubes = [tr.tube for tr in tracker._tracks.values()] + closed_all
    if linker is not None:
        for t in tubes:
            if t.class_label == "person":
                t.entity_id = linker.entity_of(t.tube_id) or t.entity_id
    cast = [CastMember(tube_ids=[t.tube_id], class_label=t.class_label, best_keyframe_ref=(t.keyframe_refs or [None])[0])
            for t in tubes]
    heights = sorted(t.max_height_px for t in tubes if t.class_label == "person" and t.max_height_px > 0)
    median_h = heights[len(heights) // 2] if heights else None
    for t in tubes:
        grade_tube(t, w, h, median_height_px=median_h, min_life_ms=int(profile.quality.min_life_s * 1000),
                   min_height_frac=profile.quality.min_height_frac, border_px=profile.quality.border_px)
        writer.write_tube(ep, t)
    writer.close(ep, CamTime(cam_utc_ms=fr.pts_ms + tick_ms), EpisodeStatus.closed, cast)
    st = reader.stats
    row = {
        "ring": "slice", "source": Path(a.source).name, "model": f"rf-detr-{a.model}", "sampled_fps": a.fps,
        "tracker": a.tracker, "detect": a.detect, "det_threshold": a.threshold, "heartbeats": heartbeats,
        "frames": frames, "decode_ms_per_frame": round(st.decode_ms_total / frames, 2),
        **{f"{k}_ms_per_frame": round(v * 1000 / frames, 2) for k, v in stage.items() if k != "detect"},
        "detect_ms_p50": round(float(np.median(detect_ms)), 2) if detect_ms else None,
        "detect_ms_p95": round(float(np.percentile(detect_ms, 95)), 2) if detect_ms else None,
        "tubes_total": len(tubes), "tubes_live_at_end": len(tracker._tracks),
        "person_tubes": sum(1 for t in tubes if t.class_label == "person"),
        "person_tubes_low_quality": sum(1 for t in tubes if t.class_label == "person" and t.quality == "low"),
        "person_dets_per_frame": round(float(np.mean(person_dets)), 2) if person_dets else 0,
        "max_concurrent_persons": max(concurrent_persons) if concurrent_persons else 0,
        "person_visibility_duty": round(state_ticks["active"] / max(1, state_ticks["active"] + state_ticks["occluded"]), 3),
        "fragmentation_est": round(sum(1 for t in tubes if t.class_label == "person") / max(1.0, float(np.mean(person_dets))), 2) if person_dets else None,
        "reid": embedder.name if embedder else "none",
        "profile": profile.name, "median_person_height_px": median_h,
        "writer": a.writer, "writer_calls": getattr(vlm_writer, "calls", 0) if vlm_writer else 0,
        "writer_ms_p50": round(float(np.median(writer_ms)), 1) if writer_ms else None,
        "tubes_described": sum(1 for t in tubes if t.attributes is not None),
        "entities": linker.entities if linker else None, "relinks": linker.relinks if linker else None,
        "merges_on_death": linker.merges if linker else None,
        "ghosts_absorbed": absorbed_total if linker else None,
        "entities_per_concurrent": round(linker.entities / max(1, max(concurrent_persons) if concurrent_persons else 1), 2) if linker else None,
        "embed_ms_p50": round(float(np.median(embed_ms)), 2) if embed_ms else None,
        "births_by_origin": dict(births_by_origin), "confirmed_by_origin": dict(confirmed_by_origin),
        "rebirths": rebirths, "duplicate_pairs": duplicate_pairs, "duplicate_frames": dup_frames,
        "debug_frames": debug_paths,
        "tubes_per_concurrent": round(sum(1 for t in tubes if t.class_label == "person") / max(1, max(concurrent_persons) if concurrent_persons else 1), 2),
        "mean_person_tube_life_s": round(float(np.mean([(t.last_seen.corrected_ms() - t.born.corrected_ms()) / 1000
                                                        for t in tubes if t.class_label == "person"] or [0])), 2),
        "merge_candidates_flagged": sum(1 for t in tubes if t.merge_candidates),
        "tubes_by_final_state": dict(Counter(t.state.value for t in tubes)),
        "classes": dict(Counter(t.class_label for t in tubes)),
        "events": dict(ev_types), "keyframes_saved": kf.saved, "episode_id": ep,
        "episode_records": sum(1 for _ in EpisodeWriter.read(writer.path(ep))),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "slice_gpu.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nepisode -> {writer.path(ep)}\nkeyframes -> {kf.root}" + (f"\ndebug frames -> {Path(a.out).parent / 'debug' / ep}" if debug_paths else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/metamorphic.py << 'EOF_VI'
"""Metamorphic tests without labels: the same clip under perturbations that must not change what
the system says. Invariants: confirmed-people count within ±1, no new event types, the whole-time
set stable. One row per variant to data/bench/metamorphic.jsonl; exit code 1 if any invariant fails.

  python bench/metamorphic.py --source /content/HI_DEF_VIDEO.mp4 --variants brightness_up,brightness_down,res540,fps6,hflip
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VARIANTS = ["brightness_up", "brightness_down", "res540", "fps6", "hflip", "shift2s"]


def make_variant(src: str, name: str, out_dir: Path) -> tuple[Path, dict]:
    """Re-encode the clip with one perturbation. fps6 and shift2s are sampling changes handled by
    slice args; the pixel variants are written with PyAV."""
    import av
    out = out_dir / f"{Path(src).stem}_{name}.mp4"
    args: dict = {}
    if name == "fps6":
        return Path(src), {"fps": 6}
    if name == "shift2s":
        return Path(src), {"skip_s": 2}
    if out.exists():
        return out, args
    cin = av.open(src); sin = cin.streams.video[0]
    cout = av.open(str(out), "w")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    w, h = sin.width, sin.height
    if name == "res540":
        w, h = (w * 540 // h) // 2 * 2, 540
    sout = cout.add_stream(codec, rate=int(sin.average_rate) if sin.average_rate else 6)
    sout.width, sout.height, sout.pix_fmt = w, h, "yuv420p"
    for fr in cin.decode(sin):
        img = fr.to_ndarray(format="rgb24").astype(np.float32)
        if name == "brightness_up":
            img = np.clip(img * 1.2, 0, 255)
        elif name == "brightness_down":
            img = np.clip(img * 0.8, 0, 255)
        elif name == "hflip":
            img = img[:, ::-1, :]
        img = img.astype(np.uint8)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        if name == "res540":
            vf = vf.reformat(width=w, height=h)
        for pkt in sout.encode(vf):
            cout.mux(pkt)
    for pkt in sout.encode():
        cout.mux(pkt)
    cout.close(); cin.close()
    return out, args


def run_slice(source: Path, out_dir: Path, base_args: list[str], fps: float, skip_s: float, zones: str | None) -> dict:
    cmd = [sys.executable, "bench/slice_gpu.py", "--source", str(source), "--out", str(out_dir / "episodes"), "--fps", str(fps)] + base_args
    if zones:
        cmd += ["--zones", zones]
    if skip_s:
        cmd += ["--skip-s", str(skip_s)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])
    txt = p.stdout
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--work", default="data/metamorphic")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("slice_args", nargs="*", help="extra args passed to slice_gpu.py after --")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    base = run_slice(Path(a.source), work / "base", a.slice_args, a.fps, 0, a.zones)
    base_people = base.get("person_tubes", 0) - base.get("person_tubes_low_quality", 0)
    rows, failures = [], []
    print(f"base: confirmed people {base_people}, entities {base.get('entities')}, events {sorted(base['events'])}")
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            src, extra = make_variant(a.source, v, work)
            r = run_slice(src, work / v, a.slice_args, extra.get("fps", a.fps), extra.get("skip_s", 0), a.zones)
        except Exception as e:
            print(f"  {v:15s} ERROR {str(e)[:120]}"); failures.append(v); continue
        people = r.get("person_tubes", 0) - r.get("person_tubes_low_quality", 0)
        new_events = sorted(set(r["events"]) - set(base["events"]))
        ok = abs(people - base_people) <= a.tolerance and not new_events
        rows.append({"variant": v, "people": people, "entities": r.get("entities"), "new_event_types": new_events, "pass": ok})
        print(f"  {v:15s} people {people:2d} (base {base_people}) entities {r.get('entities')} new events {new_events or '-'}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(v)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "metamorphic.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "metamorphic", "source": Path(a.source).name, "base_people": base_people, "rows": rows,
                            "failures": failures, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print(f"{len(rows) - len([r for r in rows if not r['pass']])}/{len(rows)} invariants hold" + (f"; failed: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
EOF_VI
cat > bench/scenario_eval.py << 'EOF_VI'
"""Score the agent against a ground-truth scenario file. Every question gets pass/fail on the
mention rules, the citation rules and the latency budget; the row goes to data/bench/scenario.jsonl.

  python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend openai
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vi.agent import FakeBackend, OpenAIBackend, TransformersBackend, ask, search_events
from vi.agent.loop import ID_RE
from vi.store import connect


import re


def _one(r, text: str) -> bool:
    r = str(r).lower()
    if r.isdigit():                       # a number must stand alone: not part of 00:14.0, E14 or 140
        return re.search(rf"(?<![\d:.\w]){re.escape(r)}(?![\d:.\w])", text) is not None
    return r in text


def _mentioned(rule, text: str) -> bool:
    """a string must appear; a list means any of its strings must appear"""
    if isinstance(rule, list):
        return any(_one(r, text) for r in rule)
    return _one(rule, text)


def score(res: dict, spec: dict, budget_ms: int, engine=None, episode_id: str | None = None) -> dict:
    f = res["final"]
    text = ID_RE.sub(" ", f.get("text") or f.get("question") or "").lower()    # ids are not numbers
    whole_ok = True
    if spec.get("whole_time_recall") and engine is not None and episode_id:
        from vi.agent import entities_present
        truth = set(entities_present(engine, episode_id, 0.9)["entity_ids"])
        cited = {c for c in f.get("citations", []) if ":E" in c}
        whole_ok = (len(cited & truth) / len(truth) >= spec["whole_time_recall"]) if truth else True
    checks = {
        "whole_time": whole_ok,
        "answered": f["action"] == "answer",
        "mentions": all(_mentioned(m, text) for m in spec.get("must_mention", []) or []),
        "avoids": not any(_one(m, text) for m in spec.get("must_not_mention", []) or []),
        "cites": (f.get("cited", False) or spec.get("expect_uncited_ok", False))
                 and (not spec.get("must_cite_prefix") or any(any(c.startswith(p) for p in spec["must_cite_prefix"]) for c in f.get("citations", []))),
        "in_budget": res["latency"]["total_ms"] <= budget_ms,
    }
    return {"q": spec["q"], "pass": all(checks.values()), "checks": checks, "latency_ms": res["latency"]["total_ms"],
            "answer": f.get("text", f.get("question", ""))[:300], "citations": f.get("citations", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--db", required=True)
    ap.add_argument("--episode", default=None, help="default: the latest episode in the store")
    ap.add_argument("--backend", choices=["fake", "openai", "transformers"], default="fake")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.scenario).read_text())
    engine = connect(a.db)
    ep = a.episode or sorted({e["episode_id"] for e in search_events(engine, limit=10000)})[-1]
    backend = {"fake": lambda: FakeBackend(), "openai": lambda: OpenAIBackend(base_url=a.base_url, model=a.model),
               "transformers": lambda: TransformersBackend(model_id=a.model)}[a.backend]()
    budget = int(spec.get("latency_budget_ms", 10000))
    results = [score(ask(engine, q["q"], ep, backend), q, budget, engine, ep) for q in spec["questions"]]
    unscored = spec.get("ground_truth", {}).get("people_total") is None
    row = {"ring": "scenario", "scenario": Path(a.scenario).name, "episode_id": ep, "backend": a.backend, "model": a.model,
           "passed": sum(r["pass"] for r in results), "total": len(results), "ground_truth_filled": not unscored,
           "latency_p50_ms": sorted(r["latency_ms"] for r in results)[len(results) // 2],
           "latency_max_ms": max(r["latency_ms"] for r in results), "results": results,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "scenario.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    for r in results:
        flags = " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["checks"].items())
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['latency_ms'] / 1000:5.1f}s  {r['q'][:70]}\n         {flags}\n         {r['answer'][:160]}")
    print(f"  {row['passed']}/{row['total']} passed, p50 {row['latency_p50_ms'] / 1000:.1f}s, max {row['latency_max_ms'] / 1000:.1f}s"
          + ("  (ground truth not filled in yet: mention rules are placeholders)" if unscored else ""))


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/scenarios/warehouse.yaml << 'EOF_VI'
# Acceptance scenario for the warehouse clip (HI_DEF_VIDEO.mp4, 17 s, 6 fps, 1280x720).
# Ground truth established by watching the clip (session 16b). bench/scenario_eval.py scores the
# agent against it; a mention rule given as a list means "any of these".
clip: HI_DEF_VIDEO.mp4
ground_truth:
  people_total: 8
  people_whole_time: 6
  people_at_table: 5
  people_right_side: 3
  people_at_conveyor: 2
  pickups: 0
  notes: |
    Table, whole clip: dark jacket + orange vest (near-left); orange vest, black hair (near-left);
    woman, orange vest over dark top (far-left end); white hard hat + orange vest + blue jeans
    (far side, moves right behind boxes ~6 s); pink/dark cap + orange vest (right of table at 0 s,
    far-middle by 8 s). Right side: yellow-green hi-vis + white hard hat at the conveyor 0-17 s;
    second hi-vis worker, dark cap, beside him at 0 s, walks along the conveyor ~8 s; dark jacket +
    orange vest walking boxes in the dock aisle top-right, first visible ~4 s, in and out after.
    Hanging jackets on the far-left racking are NOT people (false "person" detections there).
questions:
  - q: How many distinct people were in this episode, and which of them stayed the whole time?
    must_mention: [["8", "eight", "7", "seven", "9", "nine"]]     # 8, ±1 tolerance while ReID matures
    must_not_mention: ["14", "13", "12", "11"]
    must_cite_prefix: ["cam1:E", "ev_"]
    whole_time_recall: 0.8           # the cited entities must cover >= 80% of entities_present(min_coverage=0.9)
  - q: Who was at the conveyor on the right, and when did they arrive?
    must_mention: [["00:0", "start", "beginning", "0.0", "from the first", "already"]]   # the hard-hat worker is there from 0 s
    must_not_mention: []
    must_cite_prefix: ["cam1:E", "ev_"]      # any of: an entity id or the enter_zone event id
    later: ["hard hat", "hi-vis"]       # becomes must_mention once the writer VLM supplies attributes
  - q: Did anyone pick something up from a shelf?
    must_mention: [["no", "none", "nothing", "did not", "didn't", "not"]]
    must_not_mention: ["yes, "]
    must_cite_prefix: []
    expect_uncited_ok: true
latency_budget_ms: 10000
EOF_VI
cat > tests/test_profiles_generator.py << 'EOF_VI'
import json
import os
import subprocess
import sys

import pytest

from vi.generator import BRANCHES, plan
from vi.profiles import available_profiles, load_profile


def test_every_profile_loads_extends_common_and_names_known_or_declared_branches():
    names = available_profiles()
    assert "common" in names and len(names) >= 6
    common = load_profile("common")
    for n in names:
        p = load_profile(n)
        assert set(common.branches) <= set(p.branches)            # tiers add, never remove
        assert p.quality.min_height_frac == 0.4 and p.sampling["active_fps"] == 4
    t1 = load_profile("tier1_hospital")
    assert "fall" in t1.events and "enter_zone" in t1.events and "wheelchair" in t1.tube_classes


def test_plan_reports_unavailable_branches_instead_of_failing():
    p = plan(["detect", "pose", "made_up"], available_modules={"rfdetr"})
    assert p["detect"]["runnable"] and p["pose"]["runnable"] is False and "mmpose" in p["pose"]["why"]
    assert p["made_up"]["branch"] is None
    assert all(b.license and b.status in ("measured", "planned", "stub") for b in BRANCHES.values())


def test_media_zone_fires_no_events():
    from vi.events import EventCompiler, Zone
    from vi.schemas import TubeSnapshot, TubeState, Box
    z = Zone(zone_id="rack", camera_id="c1", kind="media", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    ec = EventCompiler("c1", [z], enter_ticks=1, open_grace_ms=0)
    snap = TubeSnapshot(tube_id="a", class_label="person", state=TubeState.active, box=Box(x1=10, y1=10, x2=30, y2=90))
    assert ec.on_tick([], -500) == [] and ec.on_tick([snap], 0) == [] and ec.on_tick([snap], 500) == []


def test_writer_parser_tolerates_prose_missing_cells_and_bad_colors():
    from vi.writer import parse_sheet_reply
    reply = 'Here you go:\n```json\n[{"cell_id": 0, "top_color": "ORANGE", "headwear": "white hard hat", "description": "worker at a table", "confidence": 0.9},' \
            ' {"cell_id": 1, "top_color": "neon", "carried_item": "box"}]\n```'
    r = parse_sheet_reply(reply, ["t0", "t1", "t2"])
    assert r is not None and r.expected_cells == 3 and len(r.cells) == 3
    a0, a1, a2 = (c.attributes for c in r.cells)
    assert a0.top_color.value == "orange" and "hard hat" in a0.description and a0.confidence == 0.9
    assert a1.top_color is None and a1.carried_item == "box"
    assert a2.confidence == 0.0                                      # missing cell: empty, not invented
    assert parse_sheet_reply("no json here", ["t0"]) is None


def test_pack_sheet_numbers_cells():
    import numpy as np
    from vi.writer import pack_sheet
    sheet = pack_sheet([np.zeros((120, 40, 3), np.uint8)] * 5, cell=100, cols=3)
    assert sheet.size == (300, 200)


def test_numeric_tools_and_whole_time_check(tmp_path):
    from vi.agent import ask, count_entities, coverage, entities_present, search_events
    from vi.store import connect, load_episode_file
    out = subprocess.run([sys.executable, "bench/slice_cpu.py", str(tmp_path / "ep")], capture_output=True, text=True)
    assert out.returncode == 0
    path = next((tmp_path / "ep").glob("*.jsonl"))
    engine = connect(); load_episode_file(engine, path)
    ep = search_events(engine)[0]["episode_id"]
    cov = coverage(engine, ep)
    assert cov and 0 < cov[0]["coverage"] <= 1
    assert count_entities(engine, ep)["count"] == 1
    assert entities_present(engine, ep, 0.99)["count"] == 0          # the walker is not present for the whole clip

    class Overcounter:
        name = "over"
        def __init__(self): self.n = 0
        def complete(self, messages, schema):
            self.n += 1
            if self.n == 1:
                return json.dumps({"action": "answer", "text": "There were 4 people.", "citations": ["cam1:2000:1"], "confidence": 0.9})
            assert "count_entities says 1" in messages[-1]["content"]
            return json.dumps({"action": "answer", "text": "There was 1 person.", "citations": ["cam1:2000:1"], "confidence": 0.9})
    res = ask(engine, "How many people were there?", ep, Overcounter())
    assert any(t.get("revise") == "numeric claim disagrees with tools" for t in res["trace"]) and "1 person" in res["final"]["text"]


def test_metamorphic_bench_runs_on_synthetic_with_fake_detector(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/metamorphic.py", "--source", str(clip), "--variants", "brightness_up,hflip,fps6",
                          "--fps", "5", "--work", str(tmp_path / "mm"), "--", "--model", "fake", "--detect", "frame"],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    assert "3/3 invariants hold" in out.stdout
EOF_VI
cat > SPEC.md << 'EOF_VI'
# vi-engine — frozen specification (v0.1, 2026-09-23)

This file is the canonical design. If code and this file disagree, fix one of them the same day.
Anything not in here is not settled. Sections marked **MEASURE** are open until a row in the
benchmark table (PLAN.md) fills them.

## 0. Purpose and hard constraints

A natural-language intelligence layer over video sources (live cameras, uploads, archives) that
answers open-ended questions with timestamped, evidence-backed results. Horizontal engine, not a
surveillance product.

- **C1 No training.** No fine-tuning, no LoRA, no distillation. Adaptation happens only through
  exemplar galleries, prompts, schemas, thresholds and the knowledge base.
- **C2 Open weights, commercial licenses.** Apache-2.0 / MIT / BSD / SAM License only in the
  serving path. AGPL/GPL (Ultralytics, BoxMOT, GPL YOLO forks) are reference-only.
- **C3 Compute scales with activity, not cameras × fps.** The unit of work is the object tube and
  the tube event; frames are transport.
- **C4 Everything a model could hallucinate is constrained.** Enum, nullable-with-reason, or
  confidence. Grammar-constrained decoding for every model output.
- **C5 Two-speed emission.** Geometry and events commit within ~100 ms of tick close; semantics
  arrive later as patches. No consumer blocks on the slow path.

## 0b. Build-frames (tiers)

The engine is horizontal; a deployment is a *profile* (`profiles/<tier>.yaml`) that extends `common`
and selects branches of the one generator, the event and zone vocabularies, sampling rates and quality
thresholds in scene units. `vi/generator/registry.py` lists every branch with its model, license,
cadence and status; `plan()` resolves a profile against the runtime and reports what cannot run
instead of failing. Tiers add capabilities; nothing is per-tier code. Any count, duration or
"whole time" claim in an answer comes from the store's numeric tools (`count_entities`,
`entities_present`, `coverage`), never from the model reading the script.

## 1. Topology (Block 1)

```
cameras ─► Ring 0 bitstream gate (CPU, no decode)
           └► Ring 1 selective decode + batched ROI detection (GPU)
              └► Ring 2 tube assembly + world fusion (CPU)
                 ├► Ring 3a event compiler (CPU, fast path)  ─┐
                 └► Ring 3b foveation + enrichers (GPU, slow) ─┴► episode files ─► Block 2
```

**Ring 0 — bitstream gate.** Motion vectors + macroblock metadata parsed from H.264/H.265, no pixel
reconstruction on the CPU path (libavcodec `+export_mvs` with loop-filter/IDCT skipped, or
PyNvVideoCodec 2.1 decode statistics when the camera is already on NVDEC). Per-camera adaptive
noise floor, MV-field coherence to reject PTZ/shake, luminance-step detection for scene-state.
Stationary blindness is by design and is covered by **heartbeat detections**: `HeartbeatScheduler`
adds the full frame as one more crop in the ROI batch at episode open, every 1 s on active tiles
and every 10 s on quiet tiles (`--detect hybrid`); all detections of a tick are deduplicated
class-wise, complete boxes beating crop-truncated ones (E-DET-10). Measured on real footage
in session 05: ROI-only detection lost standing people within 2.5 s regardless of tracker. Fallback for MJPEG /
intra-only streams: frame differencing at 1 fps behind the same interface (`vi/gate/base.py`).
Slice implementation: `FrameDiffGate`. Production: `MVGate` (week 3). **MEASURE:** gate FN rate,
ms per GOP per stream.

**Measured decision (sessions 06–08, one camera, L4):** full-frame detection every tick costs the
same as ROI-only (~27 ms per call; per-call overhead dominates at batch size 1–8) and tracks best
(15 tubes / 10 concurrent, life 9.0 s, 0 rebirths, vs 17/7, 5.5 s for ROI-only). The single-camera
slice therefore detects on the full frame every tick. ROI gating and heartbeat hybrids remain the
multi-camera cost lever and are re-measured when the cross-camera batching bench exists; the
hybrid duplicate-tube defect is tracked under E-DET-10.

**Ring 1 — selective decode + detect.** Decode only gated cameras at 2–5 fps sampled on NVDEC
(PyNvVideoCodec, MIT). Pack motion ROIs from many cameras into one batch; one detector forward
per batch; TensorRT FP16 on server, ONNX Runtime on edge. Detector: RF-DETR 1.7.0 Nano (edge) /
Medium–Large (server), Apache-2.0 sizes only. Client nouns via SAM 3 concept + exemplar prompts
at tube-event cadence. Every detection carries a ReID embedding. **MEASURE:** ms per packed
batch, mAP on ROIs.

**Ring 2 — tubes + fusion.** Per-camera `ByteTracker` written from the papers (dt-aware
constant-velocity Kalman, two-stage association, buffered IoU, first-tick centre gate; no code
from the ByteTrack/BoT-SORT repos, whose Kalman file traces to GPL Deep SORT), motion only on
edge, appearance-assisted on server. Measured (synthetic crossings): fragmentation 1.0 and zero ID
switches down to 6 fps; below ~3 fps motion-only association is ambiguous by construction, so
active tiles decode at ≥4 fps and R11's 2 fps floor applies to quiet tiles. Explicit lifecycle
`born → active → occluded → exited|lost → dead`. Ambiguous association records
`merge_candidates`, never silently merges. Fusion lifts tubes to world entities via homography
foot points + tile-graph transit bounds + ReID cosine; overlapping cameras merge into one
entity; best view elected for enrichment. Gallery match stamps names; misses stay anonymous
with stable ids. Slice: `SimpleIoUTracker`. **MEASURE:** HOTA/IDF1, cross-camera merge accuracy.

**Intra-camera re-linking (measured need, session 10):** on the warehouse clip frame-nano left 15
tubes for 10–11 people with zero rebirths: fragmentation comes from occlusions longer than the
2.5 s limit at the packing table. `TubeLinker` ties a newborn tube to an entity whose tube went
`lost` within 30 s and 400 px when appearance cosine ≥ 0.75, keeps a per-entity EMA + 5 exemplars,
and emits `Event(relink)` for auditability. Embedders: SigLIP 2 image tower (Apache-2.0, reliable
download) by default on GPU; OSNet (MIT) once weight hosting is verified; colour histogram on CPU.

**Ring 3a — event compiler.** Deterministic predicates over tube snapshots, zones, tile graph and
gate results; zero model calls. Tube events: enter_zone, exit_zone, dwell, loiter, approach,
meet, pickup, drop, left_behind, asset_missing_from_home, fall, run, crowd, handoff,
impossible_transition. Scene-state events: illumination_change, door_state_change,
appliance_state_change, modality_switch. Ingest events: signal_lost/restored,
camera_moved_suspect. Zone hysteresis on enter/exit. `pickup` requires the asset absent on a
heartbeat taken with nobody in the zone after a present reading; subject = visitors in between;
no visitors → `asset_missing_from_home` with no subject. Pickup/drop write the custody table.

**Ring 3b — foveation + enrichers.** Fires on tube events only (birth, quality-scored appearance
change, death). Keyframe score: area, Laplacian sharpness, low MV magnitude, occlusion-free,
exposure. Native-res crop from a per-camera ring buffer, +20% pad, macroblock-aligned; best crop
persisted at birth (`keyframe_refs`). Enhancement ladder: ≥224 px none; 96–224 px tracker-aligned
multi-frame denoise; <96 px single-image SR, `enhanced=true` propagated and confidence discounted
×0.7. Contact sheets: 12–16 crops, hard borders, cell ids, fixed `expected_cells`, one
Qwen3.5-4B call under grammar → `ContactSheetResult`. Specialists in parallel on the same crop:
pose (RTMPose-m), OCR (PP-OCRv6, vehicle/label tubes only), zero-shot tags (SigLIP 2), ReID
refresh. Each writes an `EnrichmentPatch`; schema violation → retry once → `failed=true`.
**MEASURE:** attribute accuracy, cross-cell bleed rate, ms per sheet.

**Episodes.** Episode = activity-bounded window per tile (first tube birth → tile quiet +
hysteresis), soft-cut on cast churn ≥0.6 or 30 min. File = append-only JSONL: header, ticks,
events, patches (may arrive after close), close-with-cast. Deterministic ids. Postgres 18 holds
ticks/events/entities/custody/KB; R2 holds crops and GOPs. Retrieval: BM25 (pg_search or
VectorChord-BM25) + pgvector 0.8.6, RRF fusion, rerank.

## 2. Block 2 — agent

Reasoning model (Qwen3.8-27B, fallback Qwen3.5-27B) with tools `search`, `get_script`,
`create_rule`, `run_check`, `clip`, `verify`, `inspect(hypothesis=…)`, `kb_lookup`. It never
receives video; pixels only through `verify`/`inspect` as native-res crops. Loop: ground the
question (entities, tiles, explicit time window) → resolve anchors against named-entity events
(SQL) → hybrid search → read script + verify → assemble crops/clips/naming form. Ambiguous
anchor → one clarifying question. Every claim cites entity ids, timestamps, cameras. Answers are
over world entities, never tubes. Naming form → gallery exemplars + retroactive relabel + relink.

## 3. Calibration and knowledge base

Per camera, per lighting regime: one deep read by the biggest available model → `SceneCard`
(tile hypothesis, assets → SAM 3 masks → zones, actuators with `controls: unknown`, light
sources, exits, reflective surfaces, blind regions, media zones, floor polygon, ground points,
camera pose). Everything is a hypothesis. **Movement walk** fits homography scale, derives
tile-graph edges and transit bounds, detects overlaps. **Actuation walk** toggles every switch
and door once to seed causal facts. KB = typed property graph in Postgres; every edge is a
`Fact{status, confidence, source, support, contradictions, evidence, version}`; promote at 2
supports or user confirmation, retire at 2 contradictions unless user-confirmed. Unknowns are
stored explicitly. Fact miner (backlog lane): scene-state event → actuation-like events site-wide
within ±1 s → batch adjudication by the reasoning model → hypothesis → user prompt. Every
episode records the KB version it was compiled under.

## 4. Night modality

Per-camera mode detection (chroma≈0 → IR; noise profile → low-light RGB). Mode switch is a
scene-state event. Ring 0: night threshold profile, coherence weighted higher. Ring 1: raw
frames, at most gamma/CLAHE; no per-frame learned enhancer unless the night eval shows a gap and
the license is clear. Ring 3b: enhancement per crop only. Writer schema: in non-color modalities
color fields are null with `color_reason=ir_mode`; tone fields instead; enforced by validator
and grammar. ReID: separate exemplar sets per modality; tile/time continuity weighted higher at
night. Agent states IR mode when colors are asked.

## 5. Supersedes

- Writer VLM as its own tracker → deterministic tube assembly; VLM only in Ring 3b on sheets.
- CPU frame-differencing gate → bitstream MV gate (frame-diff remains the codec fallback).
- Fixed time windows → activity episodes.
- Enhancement-before-detection → enhancement per crop, tagged.

## 6. Requirements (R1–R30, condensed)

Global: R1 no training; R2 no AGPL, BOM in CI; R3 grammar-constrained JSON everywhere; R4 two-speed
emission; R5 per-unit cost metrics per ring; R6 eval set before tuning.
Ring 0: R7 MV/decode-stats gate; R8 adaptive thresholds + coherence; R9 codec fallback; R10 heartbeats.
Ring 1: R11 NVDEC selective decode; R12 packed batches, Apache-only detector; R13 open-vocab at
event cadence; R14 ReID with every detection.
Ring 2: R15 tracker from source; R16 explicit lifecycle; R17 fusion by homography+graph+ReID;
R18 gallery match in fusion.
Ring 3a: R19 deterministic events + custody table.
Ring 3b: R20 keyframe scoring; R21 tagged enhancement ladder; R22 contact sheets with
`carried_item`; R23 parallel specialists as independent patches.
Storage: R24 episode JSONL + Postgres + R2; R25 hybrid retrieval.
Agent: R26 no video to the reasoning model; R27 clarify on ambiguity, cite everything.
Identity/edge: R28 gallery lifecycle + opt-in face + deletable; R29 rings 0–2 on edge;
R30 SAM License compliance reviewed.

## 7. Bill of materials

| Role | Pick | Version | License |
|---|---|---|---|
| Writer VLM | Qwen3.5-4B (9B if needed; 2B on edge) | HF Qwen/Qwen3.5-4B (Mar 2026) | Apache-2.0 |
| Reasoning agent | Qwen3.8-27B → fallback Qwen3.5-27B | HF Qwen/Qwen3.8-27B (Aug 2026) **verify** | Apache-2.0 (reported) |
| Detector | RF-DETR Nano (edge) / Medium–Large (server) | rfdetr 1.7.0 | Apache-2.0 |
| Open-vocab + inspect | SAM 3 / 3.1 | facebook/sam3 | SAM License |
| Counting | CountGD | HF nikigoli/CountGD | MIT |
| Tracker | BoT-SORT + ByteTrack from source | own | MIT originals |
| ReID | OSNet (torchreid); CLIP-ReID **verify license** | torchreid ckpts | MIT |
| Pose | RTMPose-m | mmpose | Apache-2.0 |
| OCR | PP-OCRv6 small/tiny | PaddleOCR 3.7.0 | Apache-2.0 |
| Tags / image emb. | SigLIP 2 base-patch16 | google/siglip2-base-patch16-224 | Apache-2.0 |
| Text emb. | Qwen3-Embedding-0.6B | HF | Apache-2.0 |
| Single-image SR | Real-ESRGAN x4plus | RealESRGAN_x4plus.pth | BSD-3 |
| GPU decode + MV stats | PyNvVideoCodec | 2.1 | MIT |
| CPU MV extraction | PyAV / libavcodec `+export_mvs` | FFmpeg 7.x | LGPL (dynamic) |
| LLM serving | vLLM + xgrammar (SGLang alternate) | ≥0.12 API, pin stable | Apache-2.0 |
| DB | PostgreSQL 18 + pgvector 0.8.6 + pg_search **verify license** / VectorChord-BM25 | — | PG / see note |
| Object storage | Cloudflare R2 | — | — |
| Eval | TrackEval | main | MIT |

Pins to verify before commit: Qwen3.8-27B card; MVTrack weights (else classical MV clustering);
pg_search license; SAM 3.1 video path in the loader; CLIP-ReID license.

## 8. Edge cases

`edge_cases.yaml` is the registry; `make coverage` fails when an implemented case has no test.
See EDGE_CASES.md (generated).
EOF_VI
cp "$0" colab/sessions/session_26_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
if [ "$HAS_GPU" = 1 ]; then python -c "import fla" 2>/dev/null || pipi flash-linear-attention >/tmp/fla.log 2>&1 || true; fi
make check
python -c "
from vi.profiles import load_profile; from vi.generator import plan, runtime_modules
p = load_profile('$PROFILE'); print('profile', p.name, '| branches:', {k: v['why'] for k, v in plan(p.branches, runtime_modules()).items()})"

step "4. database"
DB_URL="${DB_URL:-}"
if [ -z "$DB_URL" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  if ! command -v psql >/dev/null 2>&1; then (apt-get update -qq && apt-get install -y -qq postgresql) >/tmp/apt.log 2>&1 || warn "apt install postgresql failed"; fi
  if command -v psql >/dev/null 2>&1; then
    service postgresql start >/dev/null 2>&1 || true
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='vi'" | grep -q 1 || sudo -u postgres psql -qc "CREATE USER vi WITH PASSWORD 'vi';"
    sudo -u postgres psql -qc "DROP DATABASE IF EXISTS vi;" 2>/dev/null; sudo -u postgres psql -qc "CREATE DATABASE vi OWNER vi;"
    pipi "psycopg[binary]>=3.2"; DB_URL="postgresql+psycopg://vi:vi@localhost/vi"
  fi
fi
if [ -z "$DB_URL" ]; then DB_URL="sqlite+pysqlite:///data/vi.db"; rm -f data/vi.db; warn "using SQLite at data/vi.db"; fi
echo "DB_URL=$DB_URL"

step "5. slice with profile $PROFILE, SigLIP relinking and the writer VLM -> store"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes data/keyframes
  python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid siglip --profile "$PROFILE" \
    --writer qwen --writer-model "$AGENT_MODEL" --max-frames 400 > /tmp/slice_final.log 2>&1 || { warn "slice failed:"; grep -v TracerWarning /tmp/slice_final.log | tail -8; }
  grep -E "^profile|^\[writer\]|describe |relink |merge " /tmp/slice_final.log | sed 's/^/  /' | head -30
  grep -E "\"(person_tubes|person_tubes_low_quality|entities|median_person_height_px|tubes_described|writer_calls|writer_ms_p50)\"" /tmp/slice_final.log | tr -d ' ' | tr '\n' ' ' | sed 's/^/  /'; echo
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null
fi

step "6. scene script head (now with coverage and looks)"
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 14 2>&1 | grep -E "^EPISODE|^CAST|^  cam1:E|^  anon" | head -14

step "7. three questions, $AGENT_MODEL (numeric tools enforced)"
BACKEND=fake; [ "$HAS_GPU" = 1 ] && BACKEND=transformers
python bench/agent_replay.py --db "$DB_URL" data/episodes/*.jsonl --script-lines 0 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and what were they wearing?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -E "^Q:|latency:|tools:|revise|A \(|clarify|cites:"

step "8. scenario acceptance (whole-time rule now enforced)"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -E "^\s+\[(PASS|FAIL)\]|whole_time=|passed,"

step "9. metamorphic invariants on the clip (no labels): brightness, resolution, fps, flip, shift"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  python bench/metamorphic.py --source "$SOURCE" --fps 4 --zones data/zones/HI_DEF_VIDEO.json \
    --variants brightness_up,brightness_down,res540,fps6,hflip,shift2s -- --model nano --detect frame --reid siglip --profile "$PROFILE" --max-frames 400 \
    2>&1 | grep -E "^base|^  |invariants" || true
fi

step "10. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: build-frame profiles + branch registry, writer branch, numeric tools + claim checks, media zones out of events, scene-unit quality, metamorphic bench, whole-time rule"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "11. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 3, 5, 6, 7, 8, 9 back."
