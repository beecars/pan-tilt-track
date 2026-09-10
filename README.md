# pan-tilt-track

Two-DOF (pan/tilt) visual tracking head: DYNAMIXEL XL330 servo control +
YOLO26 detection/tracking, closing the loop from pixel error to servo
motion. Two-stage coarse-to-fine tracking across a fixed wide camera and
a telephoto camera mounted on the gimbal. See Architecture below.

Standalone Python package, built to be pulled in as a submodule of a
larger system (global tracking, identification).

![Hardware walkthrough](assets/hardware_diagram.gif)

## Hardware

- **Compute**: Jetson Orin Nano Super Developer Kit
  - L4T 36.5.0 / JetPack 6.2
  - No hardware video encoder: optional RTSP feed uses software
    encoding (`x264enc`), competes with YOLO for CPU
- **Cameras**: 2x IMX477 CSI sensors, `nvarguscamerasrc`, 1920x1080@30fps
  - **wide**: fixed/stationary, not mounted on the gimbal, broader FOV.
    The only camera that ever decides where to swing the gimbal to
    *acquire* a target. Mounted physically rotated 180° (corrected in
    software via `flip_method`).
  - **telephoto**: mounted on the pan/tilt bracket, narrow FOV. Runs its
    own detection and fine self-correction once a target is in its
    frame, and keeps tracking it even after it leaves wide's FOV; wide
    is only consulted again once telephoto's own lock is lost.
  - Which physical sensor is which role, and its mount orientation, is
    per-rig calibration data in `config/cameras.json` (see
    `pan_tilt_track/camera/config.py`), not hardcoded, since it changes
    on reassembly. Re-verify after any physical remount.
  - A separate per-rig calibration, `config/wide_handoff.json`, maps a
    detected pixel location in wide's (fixed) frame to an absolute
    pan/tilt goal position, necessary because wide isn't co-mounted
    with the gimbal, so a pixel offset there doesn't relate to a servo
    tick delta through a fixed gain the way telephoto's own offset does.
    Produced empirically by `scripts/calibrate_wide_handoff.py`, not
    hand-authored. See Usage below.
- **Servos**: 2x ROBOTIS DYNAMIXEL XL330, Protocol 2.0
  - Pan = ID 1, 490–3550 ticks (≈269°)
  - Tilt = ID 2, 2048–3246 ticks (≈105°, one-sided from mechanical center)
  - 57600 baud, driven directly: no external servo controller board
  - Port/baudrate/IDs/joint limits are likewise per-rig config, in
    `config/servos.json` (see `pan_tilt_track/dynamixel/config.py`)
- **Servo interface**: ROBOTIS U2D2 USB-to-TTL adapter
  - `/dev/ttyUSB0`
- **Mechanical**: 2-DOF pan/tilt bracket
  - Actuated directly by the two XL330s

## Block diagram

Two-stage coarse-to-fine visual servo. Wide (fixed) only ever fires when
telephoto currently has no lock; telephoto (on the gimbal) does its own
fine tracking otherwise, including once the target has left wide's FOV.

![Block diagram](assets/block_diagram.svg)

<sub>Source: `assets/block_diagram.mmd`. Regenerate after editing with `mermaid.ink` or `mmdc`.</sub>

The servo firmware's position PID + Profile Velocity/Acceleration is the
only closed-loop control at the joint level. The bracket physically
re-aiming the telephoto camera is what closes the loop back to vision for
fine tracking; this codebase never reads back whether a goal position
was actually reached. Wide's mapping is an open-loop coarse move (no
feedback that it actually landed telephoto on the target); telephoto's
own detector picking the target up afterward is the confirmation. See
`TrackManager` below for the seam this leaves for future ID/ReID and
true multi-camera/world-coordinate fusion, neither built yet.

## Architecture

- `dynamixel/`: servo control (`dynamixel_sdk` wrapper); torque enable,
  EEPROM/RAM writes, sync-write goal positions. No control law of its
  own. Per-rig bring-up values (port, IDs, joint limits) load from
  `config/servos.json` via `load_dynamixel_config()`.
- `tracking/`: YOLO26 `track()` wrapper, pinned to ByteTrack
  (`tracker="bytetrack.yaml"`); `target.py`'s stateless `select_target()`
  policy (closest-to-center); and `TrackManager`, which owns the
  frame-to-frame sticky lock on top of it. Wide and telephoto each run
  their own `YoloDetector`/`TrackManager` instance with independent
  track-ID spaces (their ByteTrack instances are never correlated).
  `TrackManager` is also the intended seam for future ID/ReID
  (re-acquiring a lost track by appearance) and true multi-camera/
  world-coordinate fusion, neither implemented yet. `--target-mode
  {body,head}` selects bbox center or head-keypoint centroid (head falls
  back to body center per-frame when keypoints aren't visible) for both
  cameras uniformly.
- `control/`: telephoto's fine-tracking loop (`loop.py`'s
  `TrackingLoop`, unchanged regardless of the wide-handoff stage):
  `gain.py`'s `ProportionalGain` (pixel error to goal-position delta with
  a deadband, deliberately not a full PID) and `gains.py`'s tuned
  `PAN_KP`/`TILT_KP`/`DEADBAND_PX` constants. `kp` sign is mount-specific,
  verified via `scripts/sign_check.py`: pan `kp < 0`, tilt `kp > 0` on
  this head. Re-run after any reassembly or camera reorientation. Also
  `wide_handoff.py`: the coarse stage's `WideHandoffMapper` and its
  `config/wide_handoff.json` loader/writer: an absolute pixel-to-tick
  mapping rather than an incremental gain, since wide isn't co-mounted on
  the gimbal telephoto's math assumes.
- `camera/`: `nvarguscamerasrc` capture (`gstreamer_source.py`) for
  both IMX477s, per-rig role config (`config.py`), PIP compositing
  (`pip_compositor.py`) and the RTSP-relay wrapper around it
  (`pip_relay.py`), plus an independent RTSP server (`rtsp_stream.py`)
  for remote viewing.

## Setup: Docker (recommended)

The ML stack (torch/CUDA/OpenCV) is built into an image rather than a
bare venv. See `Dockerfile` for the pinned versions and why.

```bash
docker build -t pan-tilt-track .
./scripts/docker-run.sh                        # runs the full tracker
./scripts/docker-run.sh python scripts/init_servos.py
./scripts/docker-run.sh python scripts/dual_camera_smoke_test.py
```

`docker-run.sh` passes `--runtime nvidia` for GPU, `-v /dev:/dev` +
the Argus socket for the cameras, and `--device /dev/ttyUSB0` for the
servos. GPU YOLO, dual-camera capture, and servo serial are all
confirmed working from inside the container.

Base image: `nvcr.io/nvidia/l4t-jetpack:r36.4.0`.

## Setup: bare host venv (alternative)

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`--system-site-packages` picks up the GStreamer-enabled system OpenCV.
`ultralytics` pulls in `opencv-python` transitively and can shadow it:
check with `python -c "import cv2; print(cv2.__file__)"` (should resolve
to `/usr/lib/python3.10/dist-packages`) and `pip uninstall -y
opencv-python` if not. `numpy<2` is pinned for the same reason.

YOLO runs CPU-only here unless you separately install a Jetson-native
PyTorch build. See `Dockerfile`.

## Model weights

`yolo26n.pt` / `yolo26n-pose.pt` are gitignored (`*.pt`) and are never
committed: place them at the repo root manually. `scripts/docker-run.sh`
bind-mounts them into the container automatically if present there, so no
rebuild is needed after adding or swapping a model file.

## One-time host setup

```bash
./scripts/setup_ftdi_latency.sh
```

Fixes the U2D2's FTDI latency timer (16ms default, caps round-trip rate
near 60Hz regardless of baud).

## Usage

```bash
# Bring up the servos. Profile Velocity/Acceleration are RAM and reset
# every power-cycle: run this after every reboot.
python scripts/init_servos.py

# Confirm each camera pipeline individually, or both concurrently.
python scripts/camera_smoke_test.py --role telephoto
python scripts/camera_smoke_test.py --role wide
python scripts/dual_camera_smoke_test.py

# One-time (or after any reassembly/remount): calibrate the wide-camera
# handoff mapping. Walk around, visible to both cameras for at least
# part of the session, until it collects enough samples (Ctrl+C to stop
# early, or wait for --min-samples). Writes config/wide_handoff.json,
# which run_tracker.py refuses to start without.
python scripts/calibrate_wide_handoff.py

# Full two-stage tracking loop: wide acquires, telephoto fine-tracks.
# --rtsp serves the raw (telephoto) feed at rtsp://<jetson-ip>:8554/pan-tilt.
python scripts/run_tracker.py --rtsp

# Same, with the wide camera composited in as a picture-in-picture inset.
python scripts/run_tracker.py --rtsp --rtsp-pip

# Track the head instead of the body (both cameras).
python scripts/run_tracker.py --target-mode head --rtsp

# Point at a different rig's config (defaults are config/cameras.json,
# config/servos.json, and config/wide_handoff.json).
python scripts/run_tracker.py --camera-config path/to/cameras.json --servo-config path/to/servos.json --wide-handoff-config path/to/wide_handoff.json
```

Prefix any command with `./scripts/docker-run.sh` to run in the
container.

Other scripts:

- `scripts/sign_check.py`: empirically verifies the pan/tilt `kp` sign
  convention against real hardware (telephoto's own fine-tracking loop);
  run after any reassembly.
- `scripts/read_servo_config.py`: reads current EEPROM/RAM state off
  both servos at any time (read-only, safe with torque on or off).
- `scripts/rtsp_server.py` / `scripts/dual_camera_rtsp_pip.py`:
  camera-only RTSP viewers (single feed, or wide+telephoto PIP) with no
  servo/detection dependency, for checking framing/focus cheaply.

## Tests

```bash
pytest
```

`test_gain.py`, `test_target.py`, `test_track_manager.py`, and
`test_overlay.py` are pure-logic, no hardware required. Everything else
talks to real servos and/or real cameras and is verified by running the
scripts above.
