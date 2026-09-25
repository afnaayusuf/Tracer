#!/usr/bin/env bash
# =====================================================================================
#  Tracer / vi-engine — session 11b build: SigLIP embedder fix (transformers now returns an output
#  object from get_image_features; take .pooler_output / .image_embeds), then the ReID experiment.
#  Colab: paste into one %%bash cell after the env cell, or: bash /content/build_session_11b.sh
# =====================================================================================
set -euo pipefail
GH_REPO="${GH_REPO:-afnaayusuf/Tracer}"
REPO_DIR="${REPO_DIR:-/content/$(basename "$GH_REPO")}"
SOURCE="${SOURCE:-/content/HI_DEF_VIDEO.mp4}"
SESSION="session 11b"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }
pipi() { pip install -q "$@" 2>/dev/null || pip install -q --break-system-packages "$@"; }
CLONE_URL="https://github.com/$GH_REPO.git"
[ -n "${GH_TOKEN:-}" ] && CLONE_URL="https://x-access-token:${GH_TOKEN}@github.com/$GH_REPO.git"

step "0. repository"
if [ -d "$REPO_DIR/.git" ]; then
  [ -n "${GH_TOKEN:-}" ] && git -C "$REPO_DIR" remote set-url origin "$CLONE_URL"
  git -C "$REPO_DIR" fetch -q origin
else
  git clone -q "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git config user.email >/dev/null 2>&1 || git config user.email "colab@vi-engine.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "vi-engine colab"
echo "HEAD: $(git log -1 --oneline)"

step "1. audit"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes present: $(git status --short | tr '\n' ' ')"
git pull -q --ff-only || die "local branch diverged from origin; resolve manually"
[ -n "$(git log --grep='^session 11' --format=%h)" ] || die "session 11 commit not found; run build_session_11.sh first"
EXPECTED='
.gitignore
Makefile
PLAN.md
README.md
SPEC.md
bench/README.md
bench/ring0_gate.py
bench/ring1_detect.py
bench/ring2_tubes.py
bench/slice_cpu.py
bench/slice_gpu.py
colab/README.md
colab/bootstrap.sh
edge_cases.yaml
pyproject.toml
requirements-colab.txt
tests/conftest.py
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
tests/test_tubes.py
vi/__init__.py
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
vi/tubes/__init__.py
vi/tubes/base.py
vi/tubes/bytetrack.py
vi/tubes/geometry.py
vi/tubes/kalman.py
vi/tubes/linker.py
vi/tubes/simple_iou.py
'
extras=0
for f in $(git ls-files | grep -v -e '^schemas/' -e '^data/' -e '^EDGE_CASES.md$' -e '^colab/sessions/'); do
  echo "$EXPECTED" | grep -x "$f" >/dev/null || { warn "file not in the reference tree: $f"; extras=$((extras+1)); }
done
if [ "$extras" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then die "$extras unexpected file(s); inspect, or rerun with FORCE=1"; fi
echo "audit OK"

step "2. write $SESSION files"
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


class OSNetEmbedder:
    name = "osnet"
    dim = 512

    def __init__(self, model_name: str = "osnet_x0_25", model_path: str | None = None, device: str | None = None):
        import torch
        from torchreid.utils import FeatureExtractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=model_path, device=self.device)

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
from .base import Embedder, HistogramEmbedder, crop_for_embedding, make_embedder, pick_features
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
    lk = TubeLinker("c1", sim_thr=0.75, max_gap_ms=30_000, max_jump_px=400)
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
    lk = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk.on_birth(t1, a, 0); t1.state = TubeState.lost; lk.on_close(t1, 1000)
    assert lk.on_birth(tube("c1:20000:2", 300, t=20000), a, 20000) is None            # too long ago
    lk2 = TubeLinker("c1", max_gap_ms=10_000, max_jump_px=200)
    t1 = tube("c1:0:1", 300); lk2.on_birth(t1, a, 0); t1.state = TubeState.lost; lk2.on_close(t1, 1000)
    assert lk2.on_birth(tube("c1:2000:2", 900, t=2000), a, 2000) is None                # 600 px jump
    lk3 = TubeLinker("c1")
    t1 = tube("c1:0:1", 300); lk3.on_birth(t1, a, 0); t1.state = TubeState.exited; lk3.on_close(t1, 1000)
    assert lk3.on_birth(tube("c1:2000:2", 300, t=2000), a, 2000) is None                # left through an exit


@pytest.mark.edge("E-TUBE-08")
def test_gallery_follows_appearance_drift():
    lk = TubeLinker("c1", sim_thr=0.8, ema_alpha=0.5, exemplars=3)
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
EOF_VI
cp "$0" colab/sessions/session_11b_build.sh 2>/dev/null || true
echo "wrote $(git status --porcelain | wc -l) files"

step "3. install + harness"
pipi -e ".[dev,ingest]"
make check

step "4. ReID on real footage (GPU): frame-nano | frame-nano+siglip | hybrid-tick+siglip"
cat > /tmp/_slice_row.py << 'EOF_PY'
import json, sys
label = sys.argv[1]
txt = sys.stdin.read()
if "{" not in txt:
    print(f"  {label:18s} (no row produced; see log tail below)"); sys.exit(1)
r = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
s = r['tubes_by_final_state']
ent = f"entities={r['entities']:3d} ({r['entities_per_concurrent']}/concurrent) relinks={r['relinks']:2d} embed_p50={r['embed_ms_p50']}ms" if r.get('entities') is not None else "entities=—"
print(f"  {label:18s} tubes={r['person_tubes']:3d}/{r['max_concurrent_persons']:2d}  life={r['mean_person_tube_life_s']:5.2f}s  "
      f"lost={s.get('lost',0):2d}  {ent}  reid={r['reid']}  p50={r['detect_ms_p50']}ms")
EOF_PY
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi -L | head -1)"
  pipi rfdetr==1.7.0
  python -c "import transformers" 2>/dev/null || pipi transformers
  if [ -f "$SOURCE" ]; then
    set +e
    run() { local label="$1"; shift
      python bench/slice_gpu.py --source "$SOURCE" --fps 4 --tracker byte --model nano --max-frames 400 "$@" > "/tmp/slice_$label.log" 2>&1
      grep -E "^\[reid\]|relink " "/tmp/slice_$label.log" | head -14 | sed 's/^/      /'
      python /tmp/_slice_row.py "$label" < "/tmp/slice_$label.log" || { warn "$label failed:"; grep -v TracerWarning "/tmp/slice_$label.log" | tail -8; }
    }
    run frame              --detect frame
    run frame+siglip       --detect frame  --reid siglip
    run hybrid-tick+siglip --detect hybrid --heartbeat-ms 250 --reid siglip
    set -e
    echo "  (target: entities near the number of people, 10-11; frame-nano is 15 tubes / 11 concurrent)"
  else
    warn "no clip at $SOURCE — upload one and rerun with SOURCE=/content/your.mp4"
  fi
else
  warn "no GPU in this runtime — switch to L4 and rerun"
fi

step "5. commit + push"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; else
  git commit -qm "$SESSION: unwrap transformers output objects in SigLIPEmbedder (pooler_output/image_embeds)"
fi
if [ "${NO_PUSH:-0}" != "1" ]; then git push -q && echo "pushed"; else echo "NO_PUSH=1, not pushing"; fi

step "6. summary"
echo "HEAD: $(git log -1 --oneline)"
echo "paste the three rows and the relink lines back."
