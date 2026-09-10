"""Composites an optional inset frame onto a main frame and pushes the
result to an RtspCameraServer.

Frame *acquisition* is left to the caller -- run_tracker.py and
dual_camera_rtsp_pip.py disagree about which camera is "main" and who
owns its capture session, so this only owns the composite-and-push
decision, not camera lifecycle.
"""

from __future__ import annotations

from .pip_compositor import compose_pip
from .rtsp_stream import RtspCameraServer


class PipRtspRelay:
    def __init__(self, rtsp_server: RtspCameraServer, scale: float = 0.25, margin: int = 16):
        self.rtsp_server = rtsp_server
        self.scale = scale
        self.margin = margin

    def push(self, main_frame, inset_frame=None) -> None:
        """Push `main_frame` to the RTSP server, compositing `inset_frame`
        into the bottom-left corner first if one was supplied."""
        frame = (
            main_frame
            if inset_frame is None
            else compose_pip(main_frame, inset_frame, scale=self.scale, margin=self.margin)
        )
        self.rtsp_server.push_frame(frame)
