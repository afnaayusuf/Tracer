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
    lk3 = TubeLinker("c1", sim_thr=0.75, exited_sim_thr=0.85)
    t1 = tube("c1:0:1", 300); lk3.on_birth(t1, a, 0); t1.state = TubeState.exited; lk3.on_close(t1, 1000)
    b = unit(9); b -= (b @ a) * a; b /= np.linalg.norm(b)                     # orthogonal direction
    weak = a * 0.8 + b * 0.6                                                  # cosine exactly 0.8: enough for lost, not for exited
    assert 0.75 < float(a @ weak) < 0.85
    assert lk3.on_birth(tube("c1:2000:2", 300, t=2000), weak, 2000) is None            # exited: needs stricter match
    assert lk3.on_birth(tube("c1:2500:3", 300, t=2500), a, 2500) is not None           # identical look: relinked


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
