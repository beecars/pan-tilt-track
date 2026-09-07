#!/usr/bin/env python3
"""Camera-only smoke test: confirm the nvarguscamerasrc pipeline works
before wiring in detection. Captures a few frames and saves the last one."""

import argparse
import sys

import cv2

from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default="camera_smoke_test.jpg")
    args = parser.parse_args()

    with GStreamerCameraSource() as camera:
        frame = None
        for i in range(args.frames):
            frame = camera.read()
            if frame is None:
                print(f"frame {i}: no frame received", file=sys.stderr)
                return 1
        cv2.imwrite(args.out, frame)
        print(f"Captured {args.frames} frames, saved last to {args.out} ({frame.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
