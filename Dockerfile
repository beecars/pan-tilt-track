# syntax=docker/dockerfile:1
#
# Target: Jetson Orin Nano, L4T 36.5.0 / JetPack 6.2 / CUDA 12.6.68.
# Isolates the ML stack (torch/CUDA/OpenCV) in an image: a generic `pip
# install torch` resolves a non-Tegra CUDA build on this platform, and
# opencv-python shadows the GStreamer-enabled system OpenCV that
# nvarguscamerasrc requires. Both are fixed once here instead of by hand
# on every host.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

LABEL org.opencontainers.image.title="pan-tilt-track" \
      org.opencontainers.image.description="DYNAMIXEL pan/tilt visual tracking head (YOLO26 + ByteTrack)" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# r36.4.0 (JetPack 6.1) is the closest official tag to this host's L4T
# 36.5.0 (JetPack 6.2) -- same R36 CUDA/driver ABI family.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# Non-interactive: tzdata's postinst otherwise blocks the build on an
# unanswerable debconf timezone prompt.
ENV DEBIAN_FRONTEND=noninteractive

# python3-opencv: apt build with GStreamer support, required by
# nvarguscamerasrc; must not be shadowed by pip's opencv-python.
# libopenblas0: torch's CPU BLAS backend, absent from this minimal base.
# python3-gi + gir1.2-gst-rtsp-server-1.0: GstRtspServer bindings for the
# optional RTSP viewer (camera/rtsp_stream.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv python3-opencv python3-gi gir1.2-gst-rtsp-server-1.0 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages: use the apt-installed GStreamer-enabled OpenCV
# instead of a pip wheel.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pinned early so the resolver can't drift it when installing later
# dependencies. <2: matches the NumPy ABI the apt OpenCV build and the
# Jetson-native torch wheel below were both compiled against.
RUN pip install --no-cache-dir "numpy>=1.23,<2"

# Jetson-native PyTorch build for this host's CUDA 12.6, from the
# community JetPack wheel index (pip's default PyPI wheel has no Tegra
# CUDA support). This index proxies real PyPI, so torch's declared CUDA
# deps (nvidia-cublas-cu12, nvidia-cuda-nvrtc-cu12) resolve to generic,
# wrong-CUDA-minor packages that shadow the system toolkit -- purged
# below.
RUN pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" && \
    (pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 || true)

# nvidia-cudss-cu12: the one CUDA library this torch build needs that
# JetPack's own toolkit doesn't ship. --no-deps: its declared
# cuda-toolkit[cublas] dependency would pull the mismatched packages back
# in; only its shared libraries are needed, registered via ldconfig.
RUN pip install --no-cache-dir --no-deps nvidia-cudss-cu12 && \
    find /opt/venv -path "*/nvidia/*/lib" -type d > /etc/ld.so.conf.d/nvidia-pip.conf && \
    ldconfig

# Freeze these versions so installing the project below (which pulls in
# ultralytics -> torch) can't resolve a different, non-Tegra build.
RUN pip freeze | grep -E '^(torch|torchvision|numpy)==' > /tmp/constraints.txt

WORKDIR /app

# Baked in for a standalone, no-mount `docker run`. scripts/docker-run.sh
# bind-mounts pan_tilt_track/, scripts/, and tests/ over these paths for
# day-to-day iteration, so only pyproject.toml or this Dockerfile changing
# requires a rebuild.
COPY pyproject.toml ./
COPY pan_tilt_track ./pan_tilt_track

# ultralytics declares its own opencv-python dependency; the constraints
# file pins versions but can't forbid the install, so it's removed again
# afterwards to keep the GStreamer-enabled system OpenCV in effect.
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e . && \
    pip uninstall -y opencv-python

COPY scripts ./scripts
COPY tests ./tests

# scripts/run_tracker.py disables servo torque in a `finally` block on
# KeyboardInterrupt (SIGINT); `docker stop`'s default SIGTERM would kill
# the process without running it, leaving torque enabled. Match the
# signal the app actually handles.
STOPSIGNAL SIGINT

CMD ["python", "scripts/run_tracker.py"]
