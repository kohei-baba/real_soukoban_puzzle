FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz ros-jazzy-gz-sim-vendor ros-jazzy-gz-plugin-vendor \
    ros-jazzy-gz-transport-vendor ros-jazzy-gz-tools-vendor \
    build-essential cmake pkg-config ffmpeg libegl1 libgl1-mesa-dri libglx-mesa0 \
    fonts-dejavu-core python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m venv --system-site-packages /opt/rs-venv && \
    /opt/rs-venv/bin/pip install --no-cache-dir 'torch==2.5.1' && \
    /opt/rs-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV RS_PYTHON=/opt/rs-venv/bin/python PATH=/opt/rs-venv/bin:$PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
