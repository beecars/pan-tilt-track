#!/usr/bin/env python3
"""Dual-IMX477 RTSP viewer: wide-camera feed full-frame with the
telephoto/narrow camera composited as a picture-in-picture inset in the
bottom-left corner. No servos/detection -- just the two camera feeds.

Owns both Argus capture sessions directly and pushes each composited
frame into a single RtspCameraServer, same pattern as scripts/rtsp_server.py.

Per-sensor id/orientation come from config/cameras.json (see
pan_tilt_track.camera.config) -- physical mount properties, not runtime
flags. Edit that file if the cameras are swapped or remounted.
"""

import argparse
import sys

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.camera.pip_relay import PipRtspRelay
from pan_tilt_track.camera.rtsp_stream import RtspCameraServer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--pip-scale", type=float, default=0.25, help="inset width as a fraction of main frame width")
    parser.add_argument("--pip-margin", type=int, default=16)
    parser.add_argument("--rtsp-port", default="8554")
    parser.add_argument("--mount-point", default="/pan-tilt-pip")
    args = parser.parse_args()

    cameras = load_cameras_config(args.camera_config)
    wide, telephoto = cameras.wide, cameras.telephoto

    server = RtspCameraServer(
        width=wide.capture_width,
        height=wide.capture_height,
        framerate=wide.framerate,
        port=args.rtsp_port,
        mount_point=args.mount_point,
    )
    server.start()
    print(f"RTSP server running at rtsp://<host>:{args.rtsp_port}{args.mount_point}. Ctrl+C to stop.")
    try:
        with GStreamerCameraSource(
            sensor_id=wide.sensor_id,
            capture_width=wide.capture_width,
            capture_height=wide.capture_height,
            framerate=wide.framerate,
            flip_method=wide.flip_method,
        ) as wide_cam, GStreamerCameraSource(
            sensor_id=telephoto.sensor_id,
            capture_width=telephoto.capture_width,
            capture_height=telephoto.capture_height,
            framerate=telephoto.framerate,
            flip_method=telephoto.flip_method,
        ) as telephoto_cam:
            relay = PipRtspRelay(server, scale=args.pip_scale, margin=args.pip_margin)
            while True:
                wide_frame = wide_cam.read()
                telephoto_frame = telephoto_cam.read()
                if wide_frame is None or telephoto_frame is None:
                    continue
                relay.push(wide_frame, telephoto_frame)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
