from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MV_CODECS = {"h264", "hevc", "h265", "mpeg4", "mpeg2video", "vp9", "av1"}


def gate_mode_for_codec(codec: str | None) -> Literal["mv", "framediff"]:
    """E-ING-04 / E-GATE-07: only inter-coded codecs carry motion vectors; everything else
    (MJPEG, raw, unknown) drops to the frame-difference gate behind the same interface."""
    return "mv" if (codec or "").lower() in MV_CODECS else "framediff"


class PTSFilter:
    """E-ING-07: RTSP jitter delivers repeated and reordered presentation timestamps.
    accept() keeps only strictly increasing PTS, and optionally samples to a target fps
    (R11: decode at 2–5 fps, not native)."""

    def __init__(self, target_fps: float | None = None):
        self.min_gap_ms = 1000.0 / target_fps if target_fps else 0.0
        self.last_pts_ms: int | None = None
        self.last_emit_ms: int | None = None
        self.duplicates = 0
        self.out_of_order = 0
        self.sampled_out = 0

    def accept(self, pts_ms: int) -> bool:
        if self.last_pts_ms is not None:
            if pts_ms == self.last_pts_ms:
                self.duplicates += 1
                return False
            if pts_ms < self.last_pts_ms:
                self.out_of_order += 1
                return False
        self.last_pts_ms = pts_ms
        if self.min_gap_ms and self.last_emit_ms is not None and pts_ms - self.last_emit_ms < self.min_gap_ms:
            self.sampled_out += 1
            return False
        self.last_emit_ms = pts_ms
        return True


class SizeGuard:
    """E-ING-05: a resolution change mid-stream invalidates homography and zones."""

    def __init__(self):
        self.size: tuple[int, int] | None = None
        self.changes = 0

    def check(self, width: int, height: int) -> bool:
        changed = self.size is not None and self.size != (width, height)
        if changed:
            self.changes += 1
        self.size = (width, height)
        return changed


@dataclass
class Frame:
    camera_id: str
    pts_ms: int            # stream clock (camera side)
    wall_ms: int           # host wall clock at decode
    gray: np.ndarray       # uint8 HxW, possibly downscaled by `stride`
    is_keyframe: bool
    width: int             # native width (before stride)
    height: int
    stride: int
    codec: str
    size_changed: bool = False


@dataclass
class ReaderStats:
    frames_decoded: int = 0
    frames_emitted: int = 0
    no_pts: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    sampled_out: int = 0
    size_changes: int = 0
    codec: str = ""
    gate_mode: str = ""
    native_size: tuple[int, int] | None = None
    first_pts_ms: int | None = None
    first_wall_ms: int | None = None
    decode_ms_total: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def clock_offset_ms(self) -> int | None:
        """E-ING-01 first estimate: wall clock minus stream clock at the first frame. For a
        file this is meaningless (and marked as such by the reader); for RTSP it is the
        starting point the calibration walk refines."""
        if self.first_pts_ms is None or self.first_wall_ms is None:
            return None
        return self.first_wall_ms - self.first_pts_ms


class VideoReader:
    """PyAV reader for files and RTSP URLs. Yields grayscale frames with camera-side PTS and
    host wall time. Decode happens here and only here on the CPU path; the gate never sees
    color. `stride` downsamples by integer slicing (cheap) so the gate runs on ~320–640 px."""

    def __init__(self, camera_id: str, source: str, target_fps: float | None = None,
                 max_width: int | None = 640, rtsp_tcp: bool = True):
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.max_width = max_width
        self.rtsp_tcp = rtsp_tcp
        self.stats = ReaderStats()
        self.is_live = source.startswith(("rtsp://", "rtsps://", "udp://", "srt://"))

    def frames(self) -> Iterator[Frame]:
        import av  # lazy: the harness never needs PyAV

        options = {}
        if self.is_live:
            options = {"rtsp_transport": "tcp" if self.rtsp_tcp else "udp", "stimeout": "5000000"}
        container = av.open(self.source, options=options)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            codec = stream.codec_context.name or "unknown"
            self.stats.codec = codec
            self.stats.gate_mode = gate_mode_for_codec(codec)
            tb = float(stream.time_base) if stream.time_base else None
            pts_filter = PTSFilter(self.target_fps)
            size_guard = SizeGuard()
            for frame in container.decode(stream):
                t0 = time.perf_counter()
                self.stats.frames_decoded += 1
                if frame.pts is None or tb is None:
                    self.stats.no_pts += 1
                    continue
                pts_ms = int(round(frame.pts * tb * 1000))
                if not pts_filter.accept(pts_ms):
                    continue
                wall_ms = int(time.time() * 1000)
                if self.stats.first_pts_ms is None:
                    self.stats.first_pts_ms, self.stats.first_wall_ms = pts_ms, wall_ms
                    self.stats.native_size = (frame.width, frame.height)
                size_changed = size_guard.check(frame.width, frame.height)
                gray = frame.to_ndarray(format="gray")
                stride = max(1, -(-frame.width // self.max_width)) if self.max_width else 1
                if stride > 1:
                    gray = gray[::stride, ::stride]
                self.stats.decode_ms_total += (time.perf_counter() - t0) * 1000
                self.stats.frames_emitted += 1
                yield Frame(camera_id=self.camera_id, pts_ms=pts_ms, wall_ms=wall_ms, gray=gray,
                            is_keyframe=bool(frame.key_frame), width=frame.width, height=frame.height,
                            stride=stride, codec=codec, size_changed=size_changed)
            self.stats.duplicates = pts_filter.duplicates
            self.stats.out_of_order = pts_filter.out_of_order
            self.stats.sampled_out = pts_filter.sampled_out
            self.stats.size_changes = size_guard.changes
        finally:
            container.close()
