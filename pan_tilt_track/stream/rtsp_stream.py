"""RtspCameraServer: re-serves already-captured BGR frames over RTSP via
an appsrc-based pipeline. Does not open its own camera capture session.

Usage:
    server = RtspCameraServer(width=1920, height=1080, framerate=30)
    server.start()   # runs its own GLib main loop in a background thread
    ...
    server.push_frame(bgr_frame)   # call once per captured frame
    ...
    # ... view at rtsp://<jetson-ip>:8554/pan-tilt ...
    server.stop()
"""

from __future__ import annotations

import logging
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtsp, GstRtspServer, GLib  # noqa: E402

logger = logging.getLogger(__name__)


class RtspCameraServer:
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        framerate: int = 30,
        port: str = "8554",
        mount_point: str = "/pan-tilt",
    ):
        Gst.init(None)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.mount_point = mount_point
        self._appsrc = None
        self._frame_count = 0
        self._client_count = 0
        self._last_push_time: float | None = None
        self._push_fps: float | None = None

        self._server = GstRtspServer.RTSPServer()
        self._server.set_service(port)
        self._server.connect("client-connected", self._on_client_connected)

        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(
            # block=false: push_frame() runs inline in the tracking loop
            # (and gates keyboard polling for manual/pip-swap control), so
            # a slow client or an overloaded software encoder must never
            # be able to stall it -- drop frames instead of blocking.
            "( appsrc name=source is-live=true block=false format=time "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={framerate}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            # Software encoder: the Orin Nano has no hardware encoder
            # (NVENC is Orin NX/AGX Orin only), so this competes with YOLO
            # for CPU.
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 ! "
            "h264parse ! rtph264pay name=pay0 pt=96 )"
        )

        # Allow multiple clients.
        factory.set_shared(True)
        factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
        factory.connect("media-configure", self._on_media_configure)
        self._server.get_mount_points().add_factory(mount_point, factory)

        self._loop = GLib.MainLoop()
        self._thread: threading.Thread | None = None

    def _on_media_configure(self, factory, media) -> None:
        element = media.get_element()
        self._appsrc = element.get_child_by_name("source")
        self._frame_count = 0
        media.connect("unprepared", self._on_media_unprepared)
        logger.info("RTSP client connected, appsrc ready")

    def _on_media_unprepared(self, media) -> None:
        self._appsrc = None

    def _on_client_connected(self, server, client) -> None:
        self._client_count += 1
        client.connect("closed", self._on_client_closed)

    def _on_client_closed(self, client) -> None:
        self._client_count = max(0, self._client_count - 1)

    def stats(self) -> dict:
        """Minimal live RTSP stats for display: connected client count and
        instantaneous push fps (None if nobody's currently watching)."""
        return {
            "clients": self._client_count,
            "fps": self._push_fps if self._appsrc is not None else None,
        }

    def push_frame(self, frame) -> None:
        """Feed one BGR frame (matching the constructor's width/height) to
        any connected RTSP client. No-op if nobody's watching."""
        appsrc = self._appsrc
        if appsrc is None:
            return
        now = time.monotonic()
        if self._last_push_time is not None:
            dt = now - self._last_push_time
            if dt > 0:
                self._push_fps = 1.0 / dt
        self._last_push_time = now
        duration = Gst.util_uint64_scale_int(Gst.SECOND, 1, self.framerate)
        buf = Gst.Buffer.new_wrapped(frame.tobytes())
        buf.pts = self._frame_count * duration
        buf.duration = duration
        self._frame_count += 1
        appsrc.emit("push-buffer", buf)

    def start(self) -> None:
        self._server.attach(None)
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()
        logger.info("RTSP server started at rtsp://<host>:%s%s", self._server.get_service(), self.mount_point)

    def stop(self) -> None:
        self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=2)
