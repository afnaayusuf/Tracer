from .base import Tracker
from .bytetrack import ByteTracker
from .kalman import KalmanBoxFilter
from .simple_iou import SimpleIoUTracker

TRACKERS = {"simple": SimpleIoUTracker, "byte": ByteTracker}
