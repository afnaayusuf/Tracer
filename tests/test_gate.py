import numpy as np
import pytest

from vi.gate import FrameDiffGate
from vi.schemas import Box


def _warm(gate, frame, n=8, sigma=3.0, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):   # real cameras never send identical frames: warm with sensor noise
        noisy = (frame.astype(float) + rng.normal(0, sigma, frame.shape)).clip(0, 255).astype(np.uint8)
        gate.update(noisy, i * 100)


@pytest.mark.edge("E-GATE-03")
def test_adaptive_noise_floor_ignores_sensor_noise_but_finds_object(base_frame):
    g = FrameDiffGate("c1")
    rng = np.random.default_rng(1)
    _warm(g, base_frame)
    noisy = (base_frame.astype(float) + rng.normal(0, 3, base_frame.shape)).clip(0, 255).astype(np.uint8)
    assert g.update(noisy, 600).blobs == []
    f = noisy.copy()
    f[100:160, 40:80] = 250
    r = g.update(f, 700)
    assert len(r.blobs) == 1 and r.blobs[0].box.iou(Box(x1=40, y1=100, x2=80, y2=160)) > 0.5


@pytest.mark.edge("E-GATE-02")
def test_light_switch_is_luma_step_not_motion(base_frame):
    g = FrameDiffGate("c1")
    _warm(g, base_frame)
    lit = (base_frame.astype(int) + 60).clip(0, 255).astype(np.uint8)
    r = g.update(lit, 600)
    assert r.luma_step is True and r.blobs == [] and r.global_motion is False


@pytest.mark.edge("E-GATE-04")
def test_camera_shake_is_global_motion_and_suppresses_blobs():
    rng = np.random.default_rng(2)
    textured = rng.integers(0, 255, (240, 320)).astype(np.uint8)
    g = FrameDiffGate("c1")
    _warm(g, textured)
    shifted = np.roll(textured, 12, axis=1)
    r = g.update(shifted, 600)
    assert r.global_motion is True and r.blobs == []


@pytest.mark.edge("E-GATE-01")
def test_stationary_object_fades_from_gate_by_design(base_frame):
    g = FrameDiffGate("c1", bg_alpha=0.3)
    _warm(g, base_frame)
    f = base_frame.copy()
    f[100:160, 40:80] = 250
    assert len(g.update(f, 600).blobs) == 1
    for i in range(1, 40):
        r = g.update(f, 600 + i * 100)
    assert r.blobs == []   # gate is blind to the stationary object; heartbeat detections own it


@pytest.mark.edge("E-GATE-01")
@pytest.mark.edge("E-GATE-05")
def test_heartbeat_fires_at_open_then_by_tile_state():
    from vi.gate import HeartbeatScheduler
    hb = HeartbeatScheduler(active_ms=1000, quiet_ms=5000)
    assert hb.due(0, tile_active=False) is True                 # episode open: always look once
    assert hb.due(500, tile_active=True) is False
    assert hb.due(1000, tile_active=True) is True               # active tile: every second
    assert hb.due(1999, tile_active=True) is False
    assert [hb.due(t, tile_active=False) for t in (2000, 4000, 6000, 7000)] == [False, False, True, False]
    assert hb.fired == 3
