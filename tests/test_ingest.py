import numpy as np
import pytest

from vi.events import EventCompiler
from vi.gate import FrameDiffGate
from vi.ingest import PTSFilter, SizeGuard, gate_mode_for_codec
from vi.schemas import EventType


@pytest.mark.edge("E-ING-07")
def test_pts_filter_drops_duplicates_and_reordered_frames():
    f = PTSFilter()
    seq = [0, 40, 40, 80, 60, 120, 120, 160]
    kept = [p for p in seq if f.accept(p)]
    assert kept == [0, 40, 80, 120, 160]
    assert f.duplicates == 2 and f.out_of_order == 1


@pytest.mark.edge("E-ING-03")
def test_pts_filter_samples_to_target_fps():
    f = PTSFilter(target_fps=5)                 # 200 ms gap
    kept = [p for p in range(0, 1000, 40) if f.accept(p)]   # 25 fps in
    assert kept == [0, 200, 400, 600, 800] and f.sampled_out == 20


@pytest.mark.edge("E-ING-04")
def test_codec_routes_to_gate_mode():
    assert gate_mode_for_codec("h264") == "mv" and gate_mode_for_codec("hevc") == "mv"
    assert gate_mode_for_codec("mjpeg") == "framediff"
    assert gate_mode_for_codec(None) == "framediff" and gate_mode_for_codec("rawvideo") == "framediff"


@pytest.mark.edge("E-ING-05")
def test_resolution_change_is_flagged_and_becomes_camera_moved_suspect():
    g = SizeGuard()
    assert g.check(1920, 1080) is False and g.check(1920, 1080) is False
    assert g.check(1280, 720) is True and g.changes == 1
    ec = EventCompiler("c1", [])
    evs = ec.on_size_change(5000, (1920, 1080), (1280, 720))
    assert evs[0].type == EventType.camera_moved_suspect and evs[0].payload["reason"] == "resolution_change"


@pytest.mark.edge("E-ING-07")
@pytest.mark.edge("E-GATE-02")
def test_reader_to_gate_on_synthetic_clip(tmp_path):
    av = pytest.importorskip("av")
    from vi.ingest import VideoReader
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=4, fps=10, light_switch_at_s=3)
    reader = VideoReader("c1", str(clip), target_fps=5, max_width=320)
    gate = FrameDiffGate("c1")
    results = [gate.update(fr.gray, fr.pts_ms) for fr in reader.frames()]
    st = reader.stats
    assert st.frames_decoded == 40 and st.frames_emitted == 20 and st.sampled_out == 20
    assert st.duplicates == 0 and st.out_of_order == 0 and st.codec == "mpeg4" and st.gate_mode == "mv"
    assert sum(len(r.blobs) > 0 for r in results) >= 10        # the walker is seen
    assert sum(r.luma_step for r in results) == 1              # the light switch, once
