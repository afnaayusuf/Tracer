from .base import Tracker
from .bytetrack import ByteTracker
from .kalman import KalmanBoxFilter
from .linker import TubeLinker
from .quality import grade_tube
from .simple_iou import SimpleIoUTracker

TRACKERS = {"simple": SimpleIoUTracker, "byte": ByteTracker}
