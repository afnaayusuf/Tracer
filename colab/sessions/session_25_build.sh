#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 25 build: a real ReID model, and the mirror fix.
#   * OSNet-AIN x1.0 trained on MSMT17 (torchreid model zoo, MIT) downloaded with gdown; the table
#     gets an "osnet_reid" row; the slice runs once with SigLIP and once with OSNet-ReID, and the
#     episode with the entity count closer to the truth (8) goes into the store
#   * linker merge-on-death: a tube that dies `lost` next to a live look-alike it overlapped with
#     for under a second is merged into it (the hard-hat man's third tube); long co-existence never merges
#  ONE-CELL FORM (Python cell):
#    from google.colab import userdata; import os
#    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN"); os.environ["GH_REPO"] = "afnaayusuf/Tracer"
#    !bash /content/build_session_25.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-4B}"
SESSION="session 25"
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
[ -n "$(git log --grep='^session 24' --format=%h)" ] || die "session 24 commit not found; run build_session_24.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/agent_replay.py
bench/cast_sheet.py
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
vi/harness/__init__.py
vi/harness/coverage.py
vi/harness/registry.py
vi/harness/render_edge_cases.py
vi/ingest/__init__.py
vi/ingest/reader.py
vi/ingest/synthetic.py
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
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
mkdir -p colab/sessions data/bench
cat > vi/reid/base.py << 'EOF_VI'
"""Appearance embeddings for re-identification (R14 / E-TUBE-04 / E-TUBE-08).

Backends, all open weights and no training:
  hist    CPU colour histogram (tests, fallback, weak baseline)
  siglip  google/siglip2-base-patch16-224 image tower (Apache-2.0, Hugging Face) — generic but
          reliable to download; the default GPU embedder until OSNet weight hosting is verified
  osnet   torchreid OSNet (MIT) — the dedicated ReID model in the BOM; weights come from the
          author's Google Drive, so this backend is optional and verified per runtime
Every backend returns L2-normalised float32 vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from vi.schemas import Box


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


def crop_for_embedding(frame_rgb: np.ndarray, box: Box, pad: float = 0.08) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    px, py = box.width * pad, box.height * pad
    x1, y1 = int(max(0, box.x1 - px)), int(max(0, box.y1 - py))
    x2, y2 = int(min(w, box.x2 + px)), int(min(h, box.y2 + py))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((8, 4, 3), np.uint8)
    return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])


def pick_features(out):
    """transformers has returned a bare tensor, a tuple, or an output object with pooler_output /
    image_embeds depending on version; take the pooled image embedding whichever way it arrives."""
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    return out


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


class HistogramEmbedder:
    """Upper/lower body colour histograms (8 bins per RGB channel each) -> 48-D. Deliberately
    simple: clothing colour is what survives a 13-second occlusion in a warehouse."""

    name = "hist"
    dim = 48

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            h = c.shape[0]
            parts = (c[: h // 2], c[h // 2:])
            feats = []
            for part in parts:
                for ch in range(3):
                    hist, _ = np.histogram(part[..., ch], bins=8, range=(0, 256))
                    feats.append(hist.astype(np.float32) / max(1, part.size / 3))
            out[i] = np.concatenate(feats)
        return _l2(out)


class SigLIPEmbedder:
    name = "siglip"
    dim = 768

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            feats = pick_features(self.model.get_image_features(**inputs))
        return _l2(feats.float().cpu().numpy())


OSNET_REID_WEIGHTS = {   # torchreid model zoo (docs/MODEL_ZOO.md), MSMT17 combineall=True, cross-domain rows
    "osnet_ain_x1_0": ("1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal", "osnet_ain_x1_0_msmt17.pth"),
    "osnet_x1_0": ("1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x", "osnet_x1_0_msmt17.pth"),
}


def osnet_reid_weights(model_name: str = "osnet_ain_x1_0", cache_dir: str | None = None) -> str | None:
    """Download the ReID-trained checkpoint once (gdown, Google Drive); return its path or None."""
    import os
    from pathlib import Path
    if model_name not in OSNET_REID_WEIGHTS:
        return None
    file_id, fname = OSNET_REID_WEIGHTS[model_name]
    cache = Path(cache_dir or os.environ.get("VI_WEIGHTS", Path.home() / ".cache" / "vi-weights"))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / fname
    if path.exists() and path.stat().st_size > 1_000_000:
        return str(path)
    try:
        import gdown
        gdown.download(id=file_id, output=str(path), quiet=True)
    except Exception as e:  # pragma: no cover
        print(f"[reid] osnet weights download failed ({type(e).__name__}: {str(e)[:80]})")
        return None
    return str(path) if path.exists() and path.stat().st_size > 1_000_000 else None


class OSNetEmbedder:
    """torchreid OSNet (MIT). With the ReID-trained MSMT17 checkpoint this is the purpose-built
    person embedder; without it (ImageNet weights) it is no better than a generic model."""

    name = "osnet"
    dim = 512

    def __init__(self, model_name: str = "osnet_ain_x1_0", model_path: str | None = "auto", device: str | None = None):
        import torch
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:
            from torchreid.reid.utils import FeatureExtractor      # PyPI package layout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_path == "auto":
            model_path = osnet_reid_weights(model_name)
        self.reid_trained = bool(model_path)
        self.name = "osnet_reid" if self.reid_trained else "osnet_imagenet"
        self.extractor = FeatureExtractor(model_name=model_name, model_path=model_path or "", device=self.device)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        feats = self.extractor(list(crops))
        return _l2(feats.cpu().numpy())


def make_embedder(kind: str = "auto") -> Embedder:
    """auto: osnet if importable and weights load, else siglip, else hist. Prints the choice."""
    order = ["osnet", "siglip", "hist"] if kind == "auto" else [kind]
    for k in order:
        try:
            if k == "hist":
                return HistogramEmbedder()
            if k == "siglip":
                e = SigLIPEmbedder()
            elif k == "osnet":
                e = OSNetEmbedder()
            else:
                raise ValueError(k)
            print(f"[reid] embedder: {e.name} ({e.dim}-d)")
            return e
        except Exception as ex:  # pragma: no cover
            print(f"[reid] {k} unavailable ({type(ex).__name__}: {str(ex)[:80]})")
    return HistogramEmbedder()
EOF_VI
cat > vi/reid/__init__.py << 'EOF_VI'
from .base import Embedder, HistogramEmbedder, crop_for_embedding, make_embedder, osnet_reid_weights, pick_features
EOF_VI
cat > vi/tubes/linker.py << 'EOF_VI'
"""Appearance-based re-linking of tubes into entities within one camera (E-TUBE-04), with a
drifting gallery per entity (E-TUBE-08). Cross-camera fusion reuses the same gallery format.

Rules:
  * a newborn tube is compared against entities whose last tube closed within max_gap_ms and
    whose last position is within max_jump_px; tubes that closed through an exit zone need a
    stricter similarity (exit zones may be placeholders, and people step out and back);
  * cosine similarity >= sim_thr links it to that entity, else it starts a new one;
  * each entity keeps an EMA embedding plus up to `exemplars` raw embeddings; refreshes happen
    on confident active ticks so the gallery follows jacket-on/jacket-off;
  * a link is emitted as an Event(relink) with both tube ids and the similarity, so the agent
    can cite it and a wrong link can be audited.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from vi.schemas import CamTime, Event, EventType, Tube, TubeState


@dataclass
class _Entity:
    entity_id: str
    tube_ids: list[str]
    ema: np.ndarray
    exemplars: list[np.ndarray] = field(default_factory=list)
    aux: np.ndarray | None = None
    last_box_center: tuple[float, float] = (0.0, 0.0)
    last_box_h: float = 1.0
    born_ms: int = 0
    last_active_ms: int = 0
    lost_at_ms: int | None = None      # set when its tube closed (lost/exited) OR went occluded; None while seen
    closed_as_exit: bool = False
    live_tube_id: str | None = None    # an occluded-but-live tube: linking a newborn to this entity absorbs it


class TubeLinker:
    def __init__(self, camera_id: str, sim_thr: float = 0.88, max_gap_ms: int = 30_000,
                 max_jump_px: float = 400.0, ema_alpha: float = 0.3, exemplars: int = 5,
                 exited_sim_thr: float = 0.90, near_sim_thr: float = 0.85, near_gap_ms: int = 5000,
                 near_jump_px: float = 200.0, aux_thr: float = 0.80):
        self.camera_id = camera_id
        self.sim_thr = sim_thr
        # a tube reappearing within a few seconds and a couple of body-widths of where one vanished
        # is the same person unless appearance says otherwise: the bar drops to near_sim_thr there
        self.near_sim_thr, self.near_gap_ms, self.near_jump_px = near_sim_thr, near_gap_ms, near_jump_px
        # a second, cheap appearance signal (colour histogram) must agree; it stops a generic image
        # embedding from joining a green hi-vis vest to an orange one at the same spot
        self.aux_thr = aux_thr
        self.exited_sim_thr = exited_sim_thr   # placeholder exit zones misclassify lost as exited; allow with more evidence
        self.max_gap_ms = max_gap_ms
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.n_exemplars = exemplars
        self._entities: dict[str, _Entity] = {}
        self._tube_entity: dict[str, str] = {}
        self._seq = 0
        self.relinks = 0
        self.merges = 0
        self.max_coexist_ms = 1000
        self.absorbed: list[str] = []      # ghost tubes the tracker should drop (read and clear each tick)

    # ---------------------------------------------------------------- helpers
    def _new_entity(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> _Entity:
        self._seq += 1
        ent = _Entity(entity_id=f"{self.camera_id}:E{self._seq}", tube_ids=[tube.tube_id], ema=emb.copy(),
                      exemplars=[emb.copy()], last_box_center=_center(tube), last_box_h=tube.box.height,
                      aux=None if aux is None else aux.copy(), born_ms=tube.born.corrected_ms(),
                      last_active_ms=tube.last_seen.corrected_ms())
        self._entities[ent.entity_id] = ent
        self._tube_entity[tube.tube_id] = ent.entity_id
        return ent

    @staticmethod
    def _sim(ent: _Entity, emb: np.ndarray) -> float:
        best = float(ent.ema @ emb)
        for x in ent.exemplars:
            best = max(best, float(x @ emb))
        return best

    def _candidates(self, tube: Tube, t_ms: int) -> list[_Entity]:
        cx, cy = _center(tube)
        out = []
        for e in self._entities.values():
            if e.lost_at_ms is None or t_ms - e.lost_at_ms > self.max_gap_ms:
                continue
            if ((cx - e.last_box_center[0]) ** 2 + (cy - e.last_box_center[1]) ** 2) ** 0.5 > self.max_jump_px:
                continue
            out.append(e)
        return out

    # ---------------------------------------------------------------- API
    def _thr(self, ent: _Entity, tube: Tube, t_ms: int) -> float:
        if ent.closed_as_exit:
            return self.exited_sim_thr
        cx, cy = _center(tube)
        dist = ((cx - ent.last_box_center[0]) ** 2 + (cy - ent.last_box_center[1]) ** 2) ** 0.5
        gap = t_ms - (ent.lost_at_ms or t_ms)
        return self.near_sim_thr if (gap <= self.near_gap_ms and dist <= self.near_jump_px) else self.sim_thr

    def on_birth(self, tube: Tube, emb: np.ndarray, t_ms: int, aux: np.ndarray | None = None) -> Event | None:
        cands = self._candidates(tube, t_ms)
        if aux is not None:   # second-signal gate first: candidates whose colour disagrees are out
            cands = [e for e in cands if e.aux is None or float(e.aux @ aux) >= self.aux_thr]
        if cands:
            best = max(cands, key=lambda e: self._sim(e, emb))
            sim = self._sim(best, emb)
            thr = self._thr(best, tube, t_ms)
            if sim >= thr:
                prev = best.tube_ids[-1]
                absorbed = best.live_tube_id                 # the occluded ghost this newborn replaces
                best.tube_ids.append(tube.tube_id)
                best.lost_at_ms, best.live_tube_id = None, None
                self._tube_entity[tube.tube_id] = best.entity_id
                self.on_refresh(tube, emb, aux)
                self.relinks += 1
                self.absorbed.append(absorbed) if absorbed else None
                return Event(event_id="ev_" + hashlib.sha1(f"relink|{prev}|{tube.tube_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[prev, tube.tube_id], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(sim, 3), "threshold": round(thr, 2),
                                      "gap_ms": t_ms - (tube.born.corrected_ms()), "absorbed_tube": absorbed},
                             confidence=min(1.0, sim))
        self._new_entity(tube, emb, aux)
        return None

    def on_refresh(self, tube: Tube, emb: np.ndarray, aux: np.ndarray | None = None) -> None:
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        e.ema = e.ema * (1 - self.ema_alpha) + emb * self.ema_alpha
        e.ema /= max(1e-8, np.linalg.norm(e.ema))
        if aux is not None:
            e.aux = aux.copy() if e.aux is None else (e.aux * 0.7 + aux * 0.3)
            e.aux /= max(1e-8, np.linalg.norm(e.aux))
        e.exemplars.append(emb.copy())
        if len(e.exemplars) > self.n_exemplars:
            e.exemplars.pop(0)
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height

    def on_state(self, tube: Tube, t_ms: int) -> None:
        """Called every tick for live tubes. The moment a tube goes occluded its entity becomes a
        relink candidate: the same person re-detected nearby is a newborn tube the tracker could
        not associate with the drifted prediction (the hard-hat man, session 23)."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return
        e = self._entities[eid]
        if tube.state == TubeState.occluded:
            if e.lost_at_ms is None:
                e.lost_at_ms = tube.occluded_since_ms if tube.occluded_since_ms is not None else t_ms
            e.live_tube_id = tube.tube_id
            e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        elif tube.state == TubeState.active:
            e.last_active_ms = t_ms
            if e.live_tube_id == tube.tube_id:
                e.lost_at_ms, e.live_tube_id = None, None   # seen again: no longer a candidate

    def on_close(self, tube: Tube, t_ms: int) -> Event | None:
        """A tube that dies `lost` next to a live entity it overlapped with for under a second,
        and that looks like it, was a second box on the same body: merge it (mirror of on_birth).
        Long co-existence means two people, whatever the embeddings say."""
        eid = self._tube_entity.get(tube.tube_id)
        if eid is None:
            return None
        e = self._entities[eid]
        if tube.state == TubeState.lost and len(e.tube_ids) >= 1:
            cx, cy = _center(tube)
            best, best_sim = None, 0.0
            for o in self._entities.values():
                if o is e or o.lost_at_ms is not None:
                    continue                                              # only live, currently seen entities
                overlap = min(tube.last_seen.corrected_ms(), o.last_active_ms) - max(tube.born.corrected_ms(), o.born_ms)
                dist = ((cx - o.last_box_center[0]) ** 2 + (cy - o.last_box_center[1]) ** 2) ** 0.5
                if overlap > self.max_coexist_ms or dist > self.near_jump_px:
                    continue
                sim = self._sim(o, e.ema)
                if sim > best_sim:
                    best, best_sim = o, sim
            if best is not None and best_sim >= self.near_sim_thr:
                for tid in e.tube_ids:
                    self._tube_entity[tid] = best.entity_id
                best.tube_ids = sorted(set(best.tube_ids + e.tube_ids))
                best.exemplars = (best.exemplars + e.exemplars)[-self.n_exemplars:]
                del self._entities[eid]
                self.merges += 1
                return Event(event_id="ev_" + hashlib.sha1(f"merge|{eid}|{best.entity_id}".encode()).hexdigest()[:16],
                             type=EventType.relink, t=CamTime(cam_utc_ms=t_ms), camera_id=self.camera_id,
                             subject_tube_ids=[tube.tube_id, best.tube_ids[-1]], subject_entity_ids=[best.entity_id],
                             payload={"similarity": round(best_sim, 3), "kind": "merge_on_death", "merged_entity": eid},
                             confidence=min(1.0, best_sim))
        e.last_box_center, e.last_box_h = _center(tube), tube.box.height
        e.lost_at_ms = t_ms if tube.state in (TubeState.lost, TubeState.occluded, TubeState.exited) else None
        e.closed_as_exit = tube.state == TubeState.exited
        e.live_tube_id = None
        return None

    def entity_of(self, tube_id: str) -> str | None:
        return self._tube_entity.get(tube_id)

    @property
    def entities(self) -> int:
        return len(self._entities)


def _center(t: Tube) -> tuple[float, float]:
    return ((t.box.x1 + t.box.x2) / 2.0, (t.box.y1 + t.box.y2) / 2.0)
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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--zones", default=None, help="JSON zones file; default: data/zones/<clip stem>.json if present, else edge exits + centre floor")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="tracker IoU at sampled fps")
    ap.add_argument("--max-occluded-ms", type=int, default=2500)
    ap.add_argument("--out", default="data/episodes")
    a = ap.parse_args()

    prov = Provenance(kb_version=1, pipeline_git="slice_gpu")
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
        dets = dedupe_detections([d for d in dets if is_tube_class(d.class_label)])   # E-DET-10
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
    for t in tubes:
        grade_tube(t, w, h)
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
cat > bench/reid_eval.py << 'EOF_VI'
"""Which appearance embedding actually tells these workers apart? For every embedder that loads,
embed each labelled tube's birth crop and compare same-person pairs against different-person
pairs. Prints one table and writes data/bench/reid.jsonl. Decides the linker's embedder/threshold.

  python bench/reid_eval.py data/episodes/*.jsonl --labels tests/scenarios/warehouse_identities.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from vi.episode import EpisodeWriter
from vi.reid import HistogramEmbedder
from vi.schemas.episode import TubeRecord


def load_embedders(names: list[str]) -> dict:
    out = {}
    for n in names:
        try:
            if n == "hist":
                out[n] = HistogramEmbedder()
            elif n == "siglip":
                from vi.reid.base import SigLIPEmbedder
                out[n] = SigLIPEmbedder()
            elif n == "dinov2":
                out[n] = DINOv2Embedder()
            elif n == "osnet":
                from vi.reid.base import OSNetEmbedder
                e = OSNetEmbedder(model_name="osnet_ain_x1_0")
                out[e.name] = e
                n = e.name
            print(f"[reid_eval] {n}: loaded")
        except Exception as e:
            print(f"[reid_eval] {n}: unavailable ({type(e).__name__}: {str(e)[:100]})")
    return out


class DINOv2Embedder:
    """facebook/dinov2-small (Apache-2.0): self-supervised features, known to be strong for
    instance-level similarity; CLS token, L2-normalised."""

    name = "dinov2"
    dim = 384

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, crops):
        from PIL import Image
        ims = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        with self.torch.no_grad():
            inputs = self.proc(images=ims, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        f = feats.float().cpu().numpy()
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--labels", default="tests/scenarios/warehouse_identities.yaml")
    ap.add_argument("--keyframes", default="data/keyframes")
    ap.add_argument("--embedders", default="hist,siglip,dinov2,osnet")
    ap.add_argument("--tol", type=int, default=25, help="px tolerance matching labels to birth crops")
    a = ap.parse_args()
    from PIL import Image
    labels = yaml.safe_load(Path(a.labels).read_text())
    tubes = [r.tube for r in EpisodeWriter.read(a.episode) if isinstance(r, TubeRecord) and r.tube.class_label == "person"]
    rows = []
    for t in tubes:
        ref = (t.keyframe_refs or [""])[0]
        if not ref.startswith("kf://"):
            continue
        name = Path(ref).stem                       # <born>_birth_<x>_<y>
        parts = name.split("_")
        try:
            born, x, y = int(parts[0]), int(parts[2]), int(parts[3])
        except Exception:
            continue
        lab = next((l for l in labels["tubes"] if abs(l["born_ms"] - born) <= 400 and abs(l["x"] - x) <= a.tol and abs(l["y"] - y) <= a.tol), None)
        path = Path(a.keyframes) / ref.replace("kf://", "")
        if lab and path.exists():
            rows.append({"tube": t.tube_id, "person": lab["person"], "sure": bool(lab.get("sure", True)),
                         "crop": np.asarray(Image.open(path).convert("RGB"))})
    print(f"labelled tubes with crops: {len(rows)} / {len(tubes)}  (sure: {sum(r['sure'] for r in rows)})")
    if len(rows) < 3:
        raise SystemExit("not enough labelled crops; check keyframes dir and label coordinates")
    embedders = load_embedders([e.strip() for e in a.embedders.split(",") if e.strip()])
    sure = [r for r in rows if r["sure"]]
    results = {}
    for name, emb in embedders.items():
        E = emb.embed([r["crop"] for r in rows])
        idx = {r["tube"]: i for i, r in enumerate(rows)}
        pos, neg = [], []
        for r1, r2 in itertools.combinations(sure, 2):
            s = float(E[idx[r1["tube"]]] @ E[idx[r2["tube"]]])
            (pos if r1["person"] == r2["person"] else neg).append(s)
        if not pos or not neg:
            continue
        ths = np.linspace(0.3, 0.99, 70)
        acc = [(np.mean([p >= th for p in pos]) + np.mean([n < th for n in neg])) / 2 for th in ths]
        best = float(ths[int(np.argmax(acc))])
        results[name] = {"pairs_same": len(pos), "pairs_diff": len(neg), "same_mean": round(float(np.mean(pos)), 3),
                         "same_min": round(float(np.min(pos)), 3), "diff_mean": round(float(np.mean(neg)), 3),
                         "diff_max": round(float(np.max(neg)), 3), "gap": round(float(np.min(pos) - np.max(neg)), 3),
                         "best_thr": round(best, 2), "balanced_acc": round(float(max(acc)), 3)}
        # the pairs that matter on this clip
        p4 = [r for r in sure if r["person"] == "P4"]
        results[name]["hardhat_pairs"] = [round(float(E[idx[x["tube"]]] @ E[idx[y["tube"]]]), 3) for x, y in itertools.combinations(p4, 2)]
        p7 = next((r for r in rows if r["person"] == "P7" and r["sure"]), None)
        p8 = next((r for r in rows if r["person"] == "P8"), None)
        if p7 and p8:
            results[name]["greenvest_vs_orangevest"] = round(float(E[idx[p7["tube"]]] @ E[idx[p8["tube"]]]), 3)
    print(f"\n{'embedder':8s} {'same mean/min':>14s} {'diff mean/max':>14s} {'gap':>6s} {'thr':>5s} {'bal.acc':>7s}  hard-hat pairs      green vs orange")
    for name, r in results.items():
        print(f"{name:8s} {r['same_mean']:>6.3f}/{r['same_min']:<6.3f} {r['diff_mean']:>6.3f}/{r['diff_max']:<6.3f} "
              f"{r['gap']:>6.3f} {r['best_thr']:>5.2f} {r['balanced_acc']:>7.3f}  {r['hardhat_pairs']}  {r.get('greenvest_vs_orangevest', '-')}")
    print("\nread: a positive gap means every same-person pair scores above every different-person pair; "
          "hard-hat pairs must clear the threshold, green-vs-orange must fall below it.")
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "reid.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "reid_eval", "labelled": len(rows), "results": results,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


if __name__ == "__main__":
    main()
EOF_VI
cat > tests/test_reid.py << 'EOF_VI'
import numpy as np
import pytest

from vi.reid import HistogramEmbedder, crop_for_embedding
from vi.schemas import Box, CamTime, EventType, Tube, TubeState
from vi.tubes import TubeLinker


def tube(tid, x, state=TubeState.active, t=0, h=120):
    return Tube(tube_id=tid, camera_id="c1", class_label="person", state=state, born=CamTime(cam_utc_ms=t),
                last_seen=CamTime(cam_utc_ms=t), box=Box(x1=x, y1=100, x2=x + 40, y2=100 + h))


def unit(seed, dim=48):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_histogram_embedder_is_normalised_and_separates_colours():
    e = HistogramEmbedder()
    red = np.zeros((120, 40, 3), np.uint8); red[..., 0] = 220
    blue = np.zeros((120, 40, 3), np.uint8); blue[..., 2] = 220
    v = e.embed([red, blue, red.copy()])
    assert v.shape == (3, 48) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[2] > 0.99 and v[0] @ v[1] < 0.6
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert crop_for_embedding(frame, Box(x1=100, y1=100, x2=140, y2=220)).shape[0] > 120  # padded


@pytest.mark.edge("E-TUBE-04")
def test_newborn_relinks_to_recently_lost_entity_with_matching_appearance():
    lk = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.75, max_gap_ms=30_000, max_jump_px=400)
    a = unit(1)
    t1 = tube("c1:0:1", 300)
    assert lk.on_birth(t1, a, 0) is None and lk.entities == 1
    t1.state = TubeState.lost
    lk.on_close(t1, 5000)                              # hidden behind a colleague
    t2 = tube("c1:9000:2", 340, t=9000)
    ev = lk.on_birth(t2, a * 0.95 + unit(2) * 0.05, 9000)
    assert ev is not None and ev.type == EventType.relink and ev.subject_tube_ids == ["c1:0:1", "c1:9000:2"]
    assert lk.entity_of("c1:9000:2") == lk.entity_of("c1:0:1") and lk.entities == 1 and lk.relinks == 1
    # a different-looking person at the same spot starts a new entity
    t3 = tube("c1:9500:3", 300, t=9500)
    assert lk.on_birth(t3, unit(7), 9500) is None and lk.entities == 2


@pytest.mark.edge("E-TUBE-04")
def test_relink_refuses_long_gaps_far_jumps_and_exited_tubes():
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.75, max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 1000)
    assert lk.on_birth(tube("c1:20000:2", 300, t=20000), a, 20000) is None            # too long ago
    lk2 = TubeLinker("c1", sim_thr=0.75, max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 1000)
    assert lk2.on_birth(tube("c1:2000:2", 900, t=2000), a, 2000) is None                # 600 px jump
    lk3 = TubeLinker("c1", sim_thr=0.75, exited_sim_thr=0.85)
    t1 = tube("c1:0:1", 300); lk3.on_birth(t1, a, 0); t1.state = TubeState.exited; lk3.on_close(t1, 1000)
    b = unit(9); b -= (b @ a) * a; b /= np.linalg.norm(b)                     # orthogonal direction
    weak = a * 0.8 + b * 0.6                                                  # cosine exactly 0.8: enough for lost, not for exited
    assert 0.75 < float(a @ weak) < 0.85
    assert lk3.on_birth(tube("c1:2000:2", 300, t=2000), weak, 2000) is None            # exited: needs stricter match
    assert lk3.on_birth(tube("c1:2500:3", 300, t=2500), a, 2500) is not None           # identical look: relinked


@pytest.mark.edge("E-TUBE-08")
def test_gallery_follows_appearance_drift():
    lk = TubeLinker("c1", sim_thr=0.8, near_sim_thr=0.8, ema_alpha=0.5, exemplars=3)
    base = unit(3); drift = unit(4)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, base, 0)
    steps = [(base * (1 - k) + drift * k) for k in (0.2, 0.4, 0.6, 0.8)]
    for s in steps:
        lk.on_refresh(t1, s / np.linalg.norm(s))
    t1.state = TubeState.lost; lk.on_close(t1, 5000)
    final = drift * 0.9 + base * 0.1
    ev = lk.on_birth(tube("c1:6000:2", 300, t=6000), final / np.linalg.norm(final), 6000)
    assert ev is not None                        # linked to the drifted gallery, not the birth look
    assert float(base @ (final / np.linalg.norm(final))) < 0.8      # which plain birth-embedding matching would miss


def test_pick_features_unwraps_every_transformers_return_shape():
    from types import SimpleNamespace
    from vi.reid import pick_features
    t = np.ones((2, 4))
    assert pick_features(t) is t                                                   # bare tensor
    assert pick_features((t, "aux")) is t                                          # tuple
    assert pick_features(SimpleNamespace(pooler_output=t, last_hidden_state=None)) is t   # output object (current)
    assert pick_features(SimpleNamespace(image_embeds=t)) is t                     # older naming
    class T:  # last_hidden_state only: mean-pool over tokens
        def __init__(self, a): self.a = a
        def mean(self, dim): return self.a.mean(axis=dim)
    out = pick_features(SimpleNamespace(pooler_output=None, image_embeds=None, last_hidden_state=T(np.ones((2, 5, 4)))))
    assert out.shape == (2, 4)


@pytest.mark.edge("E-TUBE-04")
def test_near_reappearance_relinks_at_the_relaxed_bar_and_colour_gate_blocks_wrong_pairs():
    a = unit(1)
    b = unit(11); b -= (b @ a) * a; b /= np.linalg.norm(b)
    weak = a * 0.7 + b * 0.714                               # cosine 0.70: under 0.75, over 0.65
    lk = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65, near_gap_ms=5000, near_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 3000)
    assert lk.on_birth(tube("c1:4000:2", 340, t=4000), weak, 4000) is not None       # 1 s, 40 px: relaxed bar applies
    lk2 = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 3000)
    assert lk2.on_birth(tube("c1:20000:2", 340, t=20000), weak, 20000) is None       # 17 s later: full bar, refused
    # colour gate: same SigLIP-ish look, different vest colour -> not linked
    green = np.zeros(48, np.float32); green[[3, 11, 19]] = 1; green /= np.linalg.norm(green)
    orange = np.zeros(48, np.float32); orange[[5, 9, 21]] = 1; orange /= np.linalg.norm(orange)
    lk3 = TubeLinker("c1", sim_thr=0.75, near_sim_thr=0.65, aux_thr=0.5)
    t1 = tube("c1:0:1", 1200); lk3.on_birth(t1, a, 0, aux=green); t1.state = TubeState.lost; lk3.on_close(t1, 4000)
    assert lk3.on_birth(tube("c1:5000:2", 1210, t=5000), a, 5000, aux=orange) is None    # identical embedding, wrong colour
    assert lk3.on_birth(tube("c1:5500:3", 1210, t=5500), a, 5500, aux=green) is not None  # same colour: linked


def test_quality_uses_the_largest_observed_height():
    from vi.tubes import grade_tube
    from vi.schemas import Box, CamTime, Tube
    t = Tube(tube_id="c1:0:9", camera_id="c1", class_label="person", born=CamTime(cam_utc_ms=0),
             last_seen=CamTime(cam_utc_ms=9700), box=Box(x1=600, y1=200, x2=620, y2=241), max_height_px=160)
    grade_tube(t, 1280, 720)
    assert t.quality == "ok"        # a 41-px final box after a 160-px life is not "tiny"


@pytest.mark.edge("E-TUBE-04")
def test_newborn_absorbs_an_occluded_live_tube_of_the_same_person():
    """The hard-hat man: his tube goes occluded at 3.0 s, he is re-detected at 3.7 s as a new
    tube while the old one is still alive; the link must happen and the ghost must be dropped."""
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.88, near_sim_thr=0.85)
    t1 = tube("c1:0:1", 700); lk.on_birth(t1, a, 0)
    t1.state = TubeState.occluded; t1.occluded_since_ms = 3000
    lk.on_state(t1, 3000)
    t2 = tube("c1:3667:2", 720, t=3667)
    ev = lk.on_birth(t2, a, 3667)
    assert ev is not None and ev.payload["absorbed_tube"] == "c1:0:1" and ev.payload["threshold"] == 0.85
    assert lk.absorbed == ["c1:0:1"] and lk.entity_of("c1:3667:2") == lk.entity_of("c1:0:1")
    # once the tracker drops the ghost, the entity is live again and not a candidate
    lk.absorbed.clear()
    t3 = tube("c1:4000:3", 720, t=4000)
    assert lk.on_birth(t3, a, 4000) is None            # nobody to link to: the entity is seen


def test_tracker_drop_returns_a_dead_tube():
    from vi.detect import Detection
    from vi.schemas import Box
    from vi.tubes import ByteTracker
    tr = ByteTracker("c1", confirm_ticks=1)
    live, _ = tr.update([Detection(box=Box(x1=0, y1=0, x2=40, y2=120), class_label="person", confidence=0.9)], 0)
    dead = tr.drop(live[0].tube_id)
    assert dead is not None and dead.state == TubeState.dead and tr.drop("nope") is None and tr._tracks == {}


@pytest.mark.edge("E-TUBE-04")
def test_dying_duplicate_merges_into_the_live_lookalike_but_long_coexistence_does_not():
    a = unit(1)
    lk = TubeLinker("c1", sim_thr=0.88, near_sim_thr=0.85)
    t1 = tube("c1:0:1", 700); lk.on_birth(t1, a, 0)                       # the person
    for t in (250, 500, 750, 1000, 1250):
        t1.state = TubeState.active; t1.last_seen = CamTime(cam_utc_ms=t); lk.on_state(t1, t)
    t2 = tube("c1:1000:2", 720, t=1000)                                     # a second box on the same body
    lk.on_birth(t2, a, 1000)                                                # t1 is active: no candidate -> new entity
    assert lk.entities == 2
    t2.state = TubeState.lost; t2.last_seen = CamTime(cam_utc_ms=1500)      # dies after 0.5 s of overlap
    ev = lk.on_close(t2, 1500)
    assert ev is not None and ev.payload["kind"] == "merge_on_death" and lk.entities == 1 and lk.merges == 1
    assert lk.entity_of("c1:1000:2") == lk.entity_of("c1:0:1")
    # two people who co-existed for 8 s are not merged however alike they look
    lk2 = TubeLinker("c1", near_sim_thr=0.85)
    p1 = tube("c1:0:1", 700); lk2.on_birth(p1, a, 0)
    p2 = tube("c1:0:2", 760); lk2.on_birth(p2, a, 0)
    for t in range(250, 8001, 250):
        p1.last_seen = CamTime(cam_utc_ms=t); p1.state = TubeState.active; lk2.on_state(p1, t)
    p2.state = TubeState.lost; p2.last_seen = CamTime(cam_utc_ms=8000)
    assert lk2.on_close(p2, 8000) is None and lk2.entities == 2
EOF_VI
cp "$0" colab/sessions/session_25_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness (+ ReID-trained OSNet weights)"
pipi -e ".[dev,ingest]"
if [ "$HAS_GPU" = 1 ]; then
  python -c "import fla" 2>/dev/null || pipi flash-linear-attention >/tmp/fla.log 2>&1 || true
  python -c "import torchreid, gdown" 2>/dev/null || pipi torchreid gdown >/tmp/torchreid.log 2>&1 || true
  python -c "from vi.reid import osnet_reid_weights; p=osnet_reid_weights('osnet_ain_x1_0'); print('   osnet_ain_x1_0 MSMT17 weights:', p or 'DOWNLOAD FAILED')"
fi
make check

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

step "5. slice twice: SigLIP vs OSNet-ReID (relink + merge-on-death); keep the one closer to 8 people"
EP_DIR=data/episodes
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  rm -rf data/episodes data/episodes_siglip data/episodes_osnet data/keyframes
  for R in siglip osnet; do
    rm -rf data/keyframes
    python bench/slice_gpu.py --source "$SOURCE" --fps 4 --model nano --tracker byte --detect frame --reid $R --max-frames 400 \
      --out data/episodes_$R > /tmp/slice_$R.log 2>&1 || { warn "slice ($R) failed:"; grep -v TracerWarning /tmp/slice_$R.log | tail -6; }
    echo "-- $R: $(grep -E '^\[reid\] embedder' /tmp/slice_$R.log | head -1)"
    grep -E "relink |merge " /tmp/slice_$R.log | sed 's/^/     /'
    grep -E "\"(person_tubes|entities|relinks|merges_on_death|ghosts_absorbed|max_concurrent_persons)\"" /tmp/slice_$R.log | tr -d ' ' | tr '\n' ' ' | sed 's/^/     /'; echo
    [ "$R" = siglip ] && cp -r data/keyframes data/keyframes_siglip 2>/dev/null || true
  done
  cp -r data/keyframes data/keyframes_osnet 2>/dev/null || true
  ES=$(grep -oE '"entities": [0-9]+' /tmp/slice_siglip.log | grep -oE '[0-9]+' || echo 99)
  EO=$(grep -oE '"entities": [0-9]+' /tmp/slice_osnet.log | grep -oE '[0-9]+' || echo 99)
  d() { echo $(( $1 > 8 ? $1 - 8 : 8 - $1 )); }
  if [ "$(d $EO)" -le "$(d $ES)" ] && grep -q "osnet_reid" /tmp/slice_osnet.log; then EP_DIR=data/episodes_osnet; CHOSEN=osnet; else EP_DIR=data/episodes_siglip; CHOSEN=siglip; fi
  echo "  chosen: $CHOSEN (siglip entities=$ES, osnet entities=$EO, truth 8)"
  rm -rf data/keyframes; cp -r data/keyframes_$CHOSEN data/keyframes 2>/dev/null || true
  python bench/cast_sheet.py $EP_DIR/*.jsonl --keyframes data/keyframes --out data/bench/cast_sheet.jpg >/dev/null || true
else
  warn "GPU or clip missing (step 0); using the CPU slice episode"
  rm -rf data/episodes; python bench/slice_cpu.py >/dev/null; EP_DIR=data/episodes
fi

step "6. ReID table with the ReID-trained OSNet"
if [ "$HAS_GPU" = 1 ] && [ -f "$SOURCE" ]; then
  python bench/reid_eval.py $EP_DIR/*.jsonl --labels tests/scenarios/warehouse_identities.yaml --keyframes data/keyframes \
    --embedders hist,siglip,osnet 2>&1 | grep -E "^embedder|^hist|^siglip|^osnet|^labelled|read:"
fi

step "7. three questions, $AGENT_MODEL"
BACKEND=fake; [ "$HAS_GPU" = 1 ] && BACKEND=transformers
python bench/agent_replay.py --db "$DB_URL" $EP_DIR/*.jsonl --script-lines 2 --backend $BACKEND --model "$AGENT_MODEL" \
  --ask "How many distinct people were in this episode, and which of them stayed the whole time?" \
  --ask "Who was at the conveyor on the right, and when did they arrive?" \
  --ask "Did anyone pick something up from a shelf?" 2>&1 | grep -E "^CAST|^Q:|latency:|A \(|clarify|cites:"

step "8. scenario acceptance (8 people, 6 whole time, 0 pickups, 10 s)"
python bench/scenario_eval.py tests/scenarios/warehouse.yaml --db "$DB_URL" --backend $BACKEND --model "$AGENT_MODEL" 2>&1 | grep -E "^\s+\[(PASS|FAIL)\]|answered=|passed,"

step "9. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: ReID-trained OSNet (MSMT17), merge-on-death, SigLIP-vs-OSNet slice comparison"
fi
if [ "${NO_PUSH:-0}" != "1" ] && [ -n "${GH_TOKEN:-}" ]; then git push -q && echo "pushed" || warn "push failed"; else warn "not pushed (NO_PUSH or no GH_TOKEN)"; fi

step "10. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste steps 5, 6, 7, 8 back."
