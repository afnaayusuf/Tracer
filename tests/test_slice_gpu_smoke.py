import json
import os
import subprocess
import sys

import pytest
from pathlib import Path


@pytest.mark.parametrize("detect", ["roi", "frame", "hybrid"])
def test_slice_runs_end_to_end_with_fake_detector(tmp_path, detect):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", detect, "--fps", "5", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".",
                         env={**os.environ, "PYTHONPATH": os.getcwd()})   # this checkout, not another editable install
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["detect"] == detect and row["frames"] > 0 and row["episode_records"] > row["frames"]
    assert row["person_tubes"] >= 1
    if detect == "hybrid":
        assert row["heartbeats"] >= 1


def test_slice_runs_with_histogram_reid(tmp_path):
    pytest.importorskip("av")
    from vi.ingest.synthetic import write_walk_clip

    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    out = subprocess.run([sys.executable, "bench/slice_gpu.py", "--source", str(clip), "--model", "fake",
                          "--detect", "frame", "--fps", "5", "--reid", "hist", "--out", str(tmp_path / "episodes")],
                         capture_output=True, text=True, cwd=".", env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-2000:]
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["reid"] == "hist" and row["entities"] >= 1 and row["embed_ms_p50"] is not None


@pytest.mark.edge("E-DET-05")
def test_media_zone_suppresses_detections_and_zone_file_is_picked_up(tmp_path):
    pytest.importorskip("av")
    import json
    from vi.ingest.synthetic import write_walk_clip
    clip = write_walk_clip(tmp_path / "walk.mp4", seconds=6, fps=10)
    zones_dir = tmp_path / "data" / "zones"; zones_dir.mkdir(parents=True)
    # a media zone covering the whole frame: every detection is suppressed -> no tubes at all
    (zones_dir / "walk.json").write_text(json.dumps([{"zone_id": "poster", "camera_id": "cam1", "kind": "media",
                                                      "polygon": [[0, 0], [320, 0], [320, 240], [0, 240]]}]))
    out = subprocess.run([sys.executable, str(Path.cwd() / "bench" / "slice_gpu.py"), "--source", str(clip), "--model", "fake",
                          "--detect", "frame", "--fps", "5", "--out", str(tmp_path / "data" / "episodes")],
                         capture_output=True, text=True, cwd=str(tmp_path), env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert out.returncode == 0, out.stderr[-1500:]
    assert "zones: ['poster'] (file" in out.stdout
    row = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    assert row["person_tubes"] == 0
