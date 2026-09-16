#!/usr/bin/env bash
# Runs the pan-tilt-track container with GPU, camera, and servo access.
#
# /dev is mounted wholesale rather than picking individual device nodes:
# nvarguscamerasrc needs a family of /dev/nvhost-* nodes that vary by
# JetPack version, plus the host's nvargus-daemon over /tmp/argus_socket.
# Matches the convention used by Jetson's own container tooling
# (jetson-containers, dusty-nv).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persists the GStreamer plugin registry across runs so a fresh container
# doesn't re-probe nvarguscamerasrc (and restart nvargus-daemon) on every
# `docker run`.
mkdir -p "$REPO_ROOT/.gst-cache"

# So run_tracker.py's 'c' key capture (written to ./captures, relative to
# the container's /app) survives past --rm instead of vanishing with the
# container. Same deal for --record's clips.
mkdir -p "$REPO_ROOT/captures"
mkdir -p "$REPO_ROOT/clips"

WEIGHT_MOUNT=()
for weights in yolo26n.pt yolo26n-pose.pt; do
    if [ -f "$REPO_ROOT/$weights" ]; then
        # Avoids re-downloading model weights into the container's
        # ephemeral filesystem on every run.
        WEIGHT_MOUNT+=(-v "$REPO_ROOT/$weights:/app/$weights")
    fi
done

docker run --rm -it \
    --runtime nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --network host \
    -v /dev:/dev \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v "$REPO_ROOT/.gst-cache:/root/.cache/gstreamer-1.0" \
    --device /dev/ttyUSB0 \
    -v "$REPO_ROOT/pan_tilt_track:/app/pan_tilt_track" \
    -v "$REPO_ROOT/scripts:/app/scripts" \
    -v "$REPO_ROOT/tests:/app/tests" \
    -v "$REPO_ROOT/config:/app/config" \
    -v "$REPO_ROOT/captures:/app/captures" \
    -v "$REPO_ROOT/clips:/app/clips" \
    "${WEIGHT_MOUNT[@]}" \
    pan-tilt-track "$@"
