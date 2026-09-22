#!/usr/bin/env bash
# Runs the pan-tilt-track container with GPU, camera, and servo access.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$REPO_ROOT/.gst-cache"
mkdir -p "$REPO_ROOT/captures"
mkdir -p "$REPO_ROOT/clips"

WEIGHT_MOUNT=()
shopt -s nullglob
for weights in "$REPO_ROOT"/*.pt "$REPO_ROOT"/*.engine; do
    WEIGHT_MOUNT+=(-v "$weights:/app/$(basename "$weights")")
done
shopt -u nullglob

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
