#!/usr/bin/env python3
"""Stand-alone RTSP viewer -- just the camera feed, no servos/detection.
Useful for checking framing/focus without paying the YOLO CPU cost.

Owns the single Argus capture session itself and pushes each frame into
RtspCameraServer directly, without a tracking loop in between.
"""

import sys

from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.camera.rtsp_stream import RtspCameraServer


def main() -> int:
    server = RtspCameraServer()
    server.start()
    print("RTSP server running. Ctrl+C to stop.")
    try:
        with GStreamerCameraSource() as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                server.push_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
