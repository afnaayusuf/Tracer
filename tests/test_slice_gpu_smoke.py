import json
import os
import subprocess
import sys

import pytest


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
