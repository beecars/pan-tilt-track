"""Composites an optional inset frame onto a main frame and pushes the
result to an RtspCameraServer."""

from __future__ import annotations

import time

from .pip_compositor import compose_pip
from .rtsp_stream import RtspCameraServer


class PipRtspRelay:
    def __init__(self, rtsp_server: RtspCameraServer, scale: float = 0.25, margin: int = 16):
        self.rtsp_server = rtsp_server
        self.scale = scale
        self.margin = margin
        self.last_compose_ms: float | None = None

    def push(self, main_frame, inset_frame=None) -> None:
        """Push `main_frame` to the RTSP server, compositing `inset_frame`
        into the bottom-left corner first if one was supplied."""
        if inset_frame is None:
            self.last_compose_ms = None
            frame = main_frame
        else:
            t0 = time.perf_counter()
            frame = compose_pip(main_frame, inset_frame, scale=self.scale, margin=self.margin)
            self.last_compose_ms = (time.perf_counter() - t0) * 1000
        self.rtsp_server.push_frame(frame)
