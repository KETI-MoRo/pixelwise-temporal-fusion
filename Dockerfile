# --------------------------------------------------
# Usage
# --------------------------------------------------
# Build the Docker image:
#
#   docker build -t c3po-pytorch1p5 .
#
# This image is intended for interactive research use
# (e.g., via `docker run` or `docker exec -it <container> /bin/bash`).

# --------------------------------------------------
# Base image
# --------------------------------------------------
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# --------------------------------------------------
# Basic environment
# --------------------------------------------------
# These ENV variables do NOT affect interactive shells (e.g., `docker exec -it <container> /bin/bash`).
# They are only used to ensure non-interactive package installation(e.g. apt-get) and stable UTF-8 behavior.
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# --------------------------------------------------
# Fix NVIDIA CUDA GPG key issue (Ubuntu 18.04)
# --------------------------------------------------
RUN apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# --------------------------------------------------
# System packages
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    vim \
    tmux \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Working directory
# --------------------------------------------------
WORKDIR /workspace

# --------------------------------------------------
# Python packages
# --------------------------------------------------
RUN pip install --upgrade pip

# Base image already includes:
# - python==3.7
# - torch==1.5.0
# - torchvision==0.6.0a0+82fd1c8
# - numpy==1.18.1
# - pillow==7.0.0
# - six==1.14.0

RUN pip install \
    future==0.18.3 \
    pycocotools==2.0.7 \
    matplotlib==3.5.3 \
    cycler==0.11.0 \
    fonttools==4.38.0 \
    python-dateutil==2.8.2 \
    pyparsing==3.1.1 \
    kiwisolver==1.4.5 \
    packaging==23.2 \
    typing-extensions==4.7.1 \
    timm==0.4.9 \
    opencv-python==4.1.2.30 \
    pandas==1.2.5

# --------------------------------------------------
# Default command
# --------------------------------------------------
CMD ["/bin/bash"]
