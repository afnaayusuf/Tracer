import numpy as np
import pytest

from vi.schemas import Box, CamTime, Provenance


@pytest.fixture
def prov():
    return Provenance(kb_version=3, pipeline_git="test")


@pytest.fixture
def t():
    def _t(ms: int, offset: int = 0) -> CamTime:
        return CamTime(cam_utc_ms=ms, offset_ms=offset)
    return _t


@pytest.fixture
def base_frame():
    rng = np.random.default_rng(0)
    return rng.normal(120, 3, (240, 320)).clip(0, 255).astype(np.uint8)


def box(x1, y1, x2, y2) -> Box:
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)
