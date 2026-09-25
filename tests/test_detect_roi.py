import numpy as np
import pytest

from vi.detect import Detection, blobs_to_rois, crop_roi, pad_batch, remap_detections
from vi.gate.base import MotionBlob
from vi.schemas import Box


def blob(x1, y1, x2, y2):
    return MotionBlob(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), energy=1.0)


def test_rois_are_padded_clamped_and_scaled_by_stride():
    rois = blobs_to_rois([blob(0, 0, 32, 48)], frame_w=1280, frame_h=720, stride=2, pad=0.25, min_side=96)
    r = rois[0]
    assert (r.x1, r.y1) == (0, 0)                     # clamped at the frame origin
    assert r.x2 >= 64 * 1.25 and r.y2 >= 96 * 1.25    # scaled by stride, then padded
    assert r.x2 <= 1280 and r.y2 <= 720


def test_small_blobs_grow_to_min_side_and_overlaps_merge():
    rois = blobs_to_rois([blob(100, 100, 108, 108)], 640, 480, stride=1, min_side=96)
    assert rois[0].x2 - rois[0].x1 >= 96 and rois[0].y2 - rois[0].y1 >= 96
    two = blobs_to_rois([blob(100, 100, 160, 200), blob(150, 150, 220, 260)], 640, 480, stride=1)
    assert len(two) == 1 and two[0].source_blobs == 2
    far = blobs_to_rois([blob(0, 0, 40, 40), blob(500, 400, 540, 440)], 640, 480, stride=1)
    assert len(far) == 2


@pytest.mark.edge("E-DET-02")
def test_remap_shifts_boxes_and_flags_frame_border_truncation():
    from vi.detect.roi import ROI
    roi = ROI(x1=100, y1=50, x2=400, y2=350)
    inside = Detection(box=Box(x1=10, y1=10, x2=60, y2=120), class_label="person", confidence=0.9)
    at_edge = Detection(box=Box(x1=0, y1=10, x2=60, y2=120), class_label="person", confidence=0.9)  # crop x=0 -> frame x=100
    bottom = Detection(box=Box(x1=10, y1=200, x2=60, y2=300), class_label="person", confidence=0.9)  # frame y2=350
    out = remap_detections([inside, at_edge, bottom], roi, frame_w=640, frame_h=352)
    assert out[0].box.x1 == 110 and out[0].box.y1 == 60 and out[0].truncated is False
    assert out[1].truncated is False                # touches the ROI edge, not the frame edge
    assert out[2].truncated is True                 # y2=350 within 2 px of frame bottom 352


def test_pad_batch_fills_with_last_crop_and_reports_real_count():
    crops = [np.zeros((10, 10, 3), np.uint8), np.ones((10, 10, 3), np.uint8)]
    padded, real = pad_batch(crops, 4)
    assert len(padded) == 4 and real == 2 and padded[3] is crops[1]
    assert pad_batch([], 4) == ([], 0)
    frame = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    from vi.detect.roi import ROI
    c = crop_roi(frame, ROI(x1=5, y1=2, x2=15, y2=12))
    assert c.shape == (10, 10, 3) and c.flags["C_CONTIGUOUS"]
