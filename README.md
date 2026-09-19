# pan-tilt-track

Often in detection and tracking we are trying to balance the tradeoff between **field-of-view (FOV)** and
**pixels-on-target (POT)**. A wide FOV camera can see more of the scene, but a target may only occupy a 
few pixels. A telephoto camera can see a target in more detail, but it may not be able to see the 
target at all if it is outside the FOV.

This repository implements a prototype multi-camera target tracking platform with pan/tilt control
to sidestep the tradeoff. A fixed position wide-angle camera acquires a target (or multiple candidates), then a telephoto 
camera on a pan/tilt bracket tracks it via PID control. 
An initial calibration step loosely maps the wide-angle camera's pixel coordinates to pan/tilt servo 
positions, enabling target handoff to the telephoto camera. 

Intended features not yet implemented: 
1. Target ID/ReID. 

## Architecture

```
pan_tilt_track/
├── dynamixel/           == ROBOTIS DYNAMIXEL SDK WRAPPER FOR SERVOS ==
│   ├── controller.py    PanTiltController: dynamixel_sdk wrapper owning
│   │                    the port/packet handlers and RAM writes (torque,
│   │                    profile velocity/acceleration, goal position).
│   └── config.py         Control-table addresses + per-rig servo config
│                        (port, IDs, joint limits/profile); loads from
│                        config/servos.json.
│
├── tracking/            == DETECTION + TRACKING LOGIC ==
│   ├── detector.py      YoloDetector: wraps Ultralytics YOLO's track()
│   │                    with ByteTrack; accepts detection or pose
│   │                    (`*-pose.pt`) models for box/keypoint output.
│   ├── target.py        select_target(): picks a detection to track,
│   │                    preferring a previously locked track ID.
│   ├── track_manager.py TrackManager: maintains a sticky lock on one
│   │                    detection's track ID across frames.
│   └── overlay.py       Draws tracking diagnostics onto a frame: boxes,
│                        locked-target highlight, crosshair, deadband
│                        boundary, current pixel-error vector.
│
├── control/             == PAN/TILT CONTROL LOOP ==
│   ├── gain.py          ProportionalGain: goal-position tick delta from
│   │                    pixel error, with a deadband.
│   ├── gains.py         Tuned pan/tilt gain constants, shared by
│   │                    run_tracker.py and sign_check.py.
│   ├── keyboard.py      NonBlockingKeyReader: non-blocking single-key
│   │                    stdin reads (cbreak mode) for interactive
│   │                    scripts.
│   ├── loop.py          TrackingLoop: wires camera -> detector ->
│   │                    target selection -> gain -> sync-write goal
│   │                    positions.
│   ├── manual.py        ManualOverride: 'm'-toggled wasd manual pan/tilt
│   │                    nudge; TrackingLoop skips its own writes while
│   │                    it's active.
│   └── wide_handoff.py  WideHandoffMapper: loads/saves wide-pixel ->
│                        goal-tick calibration (config/wide_handoff.json)
│                        and maps a pixel to an absolute goal position.
│
├── camera/                  == CAMERA CAPTURE & CONFIG ==
│   ├── source.py            CameraSource protocol: read() + release().
│   ├── gstreamer_source.py  nvarguscamerasrc capture for the IMX477;
│   │                        builds the GStreamer pipeline, converts
│   │                        frames to BGR numpy arrays.
│   └── config.py             Per-rig camera config: sensor id, capture
│                            mode, mount orientation (wide/telephoto).
│
├── stream/               == RTSP STREAMING & RECORDING ==
│   ├── clip_recorder.py  ClipRecorder: writes frames into a fixed-length
│   │                     local MP4 clip for headless recording
│   │                     (--record in run_tracker.py).
│   ├── pip_compositor.py  Composites a smaller inset frame onto a larger
│   │                      main frame (picture-in-picture).
│   ├── pip_relay.py       Composites an optional inset onto a main frame
│   │                      and pushes the result to an RtspCameraServer.
│   └── rtsp_stream.py     RtspCameraServer: re-serves already-captured
│                          BGR frames over RTSP via an appsrc pipeline.
│
└── ui/               == CLI-BASED DASHBOARD ==
    └── dashboard.py  LiveDashboard / PlainReporter: rich-based live CLI
                       status view for run_tracker.py (acquire/handoff
                       state, detection counts, pan/tilt position); the
                       --plain fallback just prints lines.
```

`TrackManager`'s independent per-camera instances are never ID-correlated
across cameras. To be addressed in future work (ID/ReID). 
`--mode {body,head,animal}` selects what both cameras aim at and detect:
bbox center (body, any class), head-keypoint centroid (head, needs a pose
model, person only), or bbox center restricted to COCO bird/cat/dog
classes (animal). An explicit `--classes` overrides whatever classes
`--mode` would otherwise pick. `kp` sign is mount-specific, verified via
`scripts/sign_check.py`: pan `kp < 0`, tilt `kp > 0` on this head;
re-run after any reassembly or camera reorientation.

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

### TensorRT engine (optional)

`--model` accepts a `.engine` file, so `YoloDetector` is TensorRT-compatible
out of the box. Exporting one is left up to the user (`yolo export
model=yolo26n.pt format=engine imgsz=H,W device=0`, run on-device since
engines aren't portable across hardware/JetPack/TensorRT versions) -- match
`imgsz` to the camera's aspect ratio rather than a plain square value, and
raise `--det-conf` alongside a higher imgsz to compensate for the extra
low-confidence detections that come with it.

```bash
./scripts/docker-run.sh python scripts/run_tracker.py --model yolo26n.engine --det-conf 0.45
```

## Enclosure Design

Most parts are 3D printed, with the exception of **`DYNAMIXEL H101`** and **`S102`** servo brackets
and various assembly hardware (M2 and M3 screws, nuts, spacers, heat-set inserts, etc.). The model 
files can be found in `assets/pan-tilt-track.step`. Construction details are not included in this repository, but the 
short video below shows the basic assembly and mechanical operation. 

<p align="center">
  <img src="assets/hardware_diagram.webp" alt="Hardware walkthrough" width="480">
</p>

## Electronics & Hardware

### Compute

Jetson Orin Nano Super Developer Kit. `L4T 36.5.0` / `JetPack 6.2`. `MAXN` power mode. 

### Servos

2x **`ROBOTIS DYNAMIXEL XL330`** w/ **`ROBOTIS U2D2`** USB-to-TTL adapter (assumed to be on 
`/dev/ttyUSB0`).


| Joint | ID | Range |
| --- | --- | --- |
| Pan | 1 | ≈269° |
| Tilt | 2 | ≈105°, one-sided from mechanical center |

Port/baudrate/IDs/joint limits are in`config/servos.json` (also see 
`pan_tilt_track/dynamixel/config.py`).

#### Position PID / profile tuning

A PID profile is written to the servo's RAM control table by `PanTiltController.initialize()` on 
**every startup**.

| Joint | profile_velocity | profile_acceleration | position_p_gain | position_i_gain |
| --- | --- | --- | --- | --- |
| Pan | 200 | 30 | 400 (default) | 30 |
| Tilt | 200 | 30 | 800 | 60 |

>*Note: If other servos or cameras are used, the above values may need to be re-tuned. The 
`scripts/init_servos.py` script can be used to write new values to the servos' RAM. For the listed hardware, integral gain is needed to counteract gravity on the tilt axis. It was also found to help with friction and/or larger moment on the pan axis.*

### Cameras
This project was validated with (2x) **`Arducam IMX477`** CSI sensors. Other 
Jetson-compatible cameras may also work, but it is important to check with the manufacturer if the
camera modules can be used in a dual-CSI configuration. The Arducam IMX477 MINI does provide dual-CSI
support for Jetson Orin, and the platform allows for lenses to be swapped so that one sensor can be
equipped for wide-angle target acquisition (see more) and the other for telephoto (see "better"). 

| Lens | Mount | Role in tracking
| --- | --- | --- |
| **wide** | Fixed/stationary | Decides where to swing the gimbal to *acquire* a target. |
| **telephoto** | Pan/Tilt | Tracks targets (via PID control loop) once acquired. Higher pixels-on-target. 

Relevant specs are defined in `config/cameras.json` (used for `gstreamer` pipelines and calibration).

#### Calibration

Wide and telephoto are non-collocated cameras, so a detection in wide's view corresponds to a "line" 
of possible positions projected onto the telephoto camera's sensor. Unlike many vision tasks that 
require precise correspondence (e.g. traditional stereo matching), this system just needs to put a 
target somewhere in the telephoto camera's FOV. If the calibration is performed at or near the 
"expected" target distance, the mapping of pixel-to-pan/tilt angle is more than accurate enough. 

A run-once calibration `scripts/calibrate_wide_handoff.py` produces `config/wide_handoff.json`, which
maps a detected pixel location in the wide camera's fixed frame to an absolute pan/tilt
"goal" position. 

**Notably, this calibration can be run in-situ without any special calibration 
target. It uses a keypoint regression to find correspondeces between detections in the two cameras' 
views. The system can be placed in a real environment, loaded with a detection model, and the 
calibration can be performed using the targets of interest at their "expected" distance(s).**

During calibration, `scripts/calibrate_wide_handoff.py` redraws a live status block in place
(servo state, telephoto's centering error, and an ASCII map of where in the wide frame samples
have landed). Example partway through a run:

```
Wide-handoff calibration -- Ctrl+C to stop early

servo:    SETTLED
center:   pan  ----------#----------     +2.3px (deadband +/-6px)
          tilt ----------#----------     -1.1px
wide:     1 detection -> sample recorded
samples:  43/100  [========------------]

coverage (wide frame):
  ...............|...............
  ...............|...............
  ...............|...............
  ...............|o..o.........o.
  .ooooo...o.....|.....o..oo...o.
  oo..o........o.|oo.oo.....o....
  -#---o-o-o-o--o+----o-oo-------
  .o.......o..o..o...............
  ...............|..oo.....o.....
  ...............|...............
  ...............|...............
  ...............|...............
  ...............|...............
```

`center` shows telephoto's current pixel error on each axis as a bar with `|` at zero and `#` at
the current value; a sample is only recorded when both axes are within the deadband and the wide view
sees exactly one detection. The coverage map buckets recorded samples
by their position in the wide frame (`o` = 1-2 samples in that cell, `#` = 3+). This gives a quick 
visual check on the distribution of samples across the wide frame. The calibration will stop 
automatically once the minimum number of samples is reached, or can be stopped early with Ctrl+C.

## Usage

The commands below would be executed via `./scripts/docker-run.sh <command>` from the host, or 
otherwise directly from inside the (properly initialized) container. 

```bash
# Bring up the servos. Profile Velocity/Acceleration are RAM and reset
# every power-cycle: run this after every reboot.
python scripts/init_servos.py

# Confirm each camera pipeline individually, or both concurrently.
python scripts/camera_smoke_test.py --role telephoto
python scripts/camera_smoke_test.py --role wide
python scripts/dual_camera_smoke_test.py

# One-time (or after any reassembly/remount) -- see Calibration below.
python scripts/calibrate_wide_handoff.py

# Full two-stage tracking loop: wide acquires, telephoto fine-tracks.
python scripts/run_tracker.py --rtsp

# See all options (--mode, --rtsp-pip, --model, --camera-config, ...).
python scripts/run_tracker.py --help
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
- `scripts/setup_ftdi_latency.sh`: one-time fix for the U2D2's FTDI
  latency timer (defaults to 16ms, capping round-trip rate near 60Hz);
  installs a udev rule so it persists across reboots/replugs and adds
  the invoking user to `dialout`.

## Tests

```bash
pytest
```

`test_gain.py`, `test_target.py`, `test_track_manager.py`, and
`test_overlay.py` are pure-logic, no hardware required. Everything else
talks to real servos and/or real cameras and is verified by running the
scripts above.
