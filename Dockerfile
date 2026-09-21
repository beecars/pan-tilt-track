# syntax=docker/dockerfile:1
#
# ML stack (torch/CUDA/OpenCV) for Jetson Orin Nano, L4T 36.5.0 / JetPack
# 6.2 / CUDA 12.6.68. Generic pip torch lacks Tegra CUDA support; pip
# opencv-python shadows the GStreamer-enabled system OpenCV.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

LABEL org.opencontainers.image.title="pan-tilt-track" \
      org.opencontainers.image.description="DYNAMIXEL pan/tilt visual tracking head (YOLO26 + ByteTrack)" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# r36.4.0 (JetPack 6.1): closest official tag to this host's JetPack 6.2, same R36 ABI family.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# Prevents tzdata postinst from blocking on debconf timezone prompt.
ENV DEBIAN_FRONTEND=noninteractive

# python3-opencv: GStreamer support for nvarguscamerasrc. libopenblas0: torch
# BLAS backend. python3-gi/gir1.2-gst-rtsp-server-1.0: RTSP viewer bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv python3-opencv python3-gi gir1.2-gst-rtsp-server-1.0 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages: exposes apt's GStreamer-enabled OpenCV to the venv.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# <2: ABI floor shared by the apt OpenCV build and the torch wheel below.
RUN pip install --no-cache-dir "numpy>=1.23,<2"

# Jetson-native torch build. Index proxies PyPI, so torch's declared CUDA
# deps resolve to generic (wrong CUDA minor) packages; purged below.
RUN pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" && \
    (pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 || true)

# CUDA lib torch requires, absent from JetPack. --no-deps: its cuda-toolkit
# dependency would reintroduce the packages purged above.
RUN pip install --no-cache-dir --no-deps nvidia-cudss-cu12 && \
    find /opt/venv -path "*/nvidia/*/lib" -type d > /etc/ld.so.conf.d/nvidia-pip.conf && \
    ldconfig

# Constrains the ultralytics install below from resolving a non-Tegra torch.
RUN pip freeze | grep -E '^(torch|torchvision|numpy)==' > /tmp/constraints.txt

# For `yolo export format=engine`; TensorRT is in the base image via apt,
# onnx is not. Constrained: onnxslim's numpy floor would otherwise bump to 2.x.
RUN pip install --no-cache-dir -c /tmp/constraints.txt onnx onnxslim && \
    pip freeze | grep -E '^(onnx|onnxslim)==' >> /tmp/constraints.txt

WORKDIR /app

# Enables standalone `docker run`; docker-run.sh bind-mounts over these for iteration.
COPY pyproject.toml ./
COPY pan_tilt_track ./pan_tilt_track

# ultralytics declares opencv-python; removed to keep the GStreamer build authoritative.
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e . && \
    pip uninstall -y opencv-python

COPY scripts ./scripts
COPY tests ./tests
COPY config ./config

# run_tracker.py disables servo torque on SIGINT, not the default SIGTERM.
STOPSIGNAL SIGINT

CMD ["python", "scripts/run_tracker.py"]
