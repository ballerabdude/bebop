"""Runtime + teacher settings for the bebop robot.

Default video source is the firmware's MJPEG endpoint — the firmware
owns the camera and bebop-vision never opens a capture device directly
(see `firmware/bebop-linux/src/video.rs` for the ownership split).
"""

DEFAULT_SOURCE = "http://bebop.local:9090/video"
DEFAULT_NAV_MODEL = "weights/navseg"
DEFAULT_CONFIDENCE = 0.5
DEVICE = "auto"
RECORD_CONCEPTS = ("floor", "wall", "person", "chair", "table", "door")