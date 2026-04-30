# syntax=docker/dockerfile:1.6
#
# Template Dockerfile for the Bebop robot application.
# This is the image that `bebop-agent` pulls + runs on the Jetson.
#
# Base image: pick the NVIDIA L4T image that matches the JetPack version
# installed on your robots. Examples:
#
#   * nvcr.io/nvidia/l4t-base:r36.3.0
#   * nvcr.io/nvidia/l4t-jetpack:r36.3.0
#   * nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3
#   * nvcr.io/nvidia/l4t-ml:r36.2.0-py3
#
# Build on a Jetson or with `docker buildx` for arm64:
#
#   docker buildx build \
#       --platform linux/arm64 \
#       -t your-registry/bebop-app:dev \
#       -f docker/robot-app/Dockerfile .
#

ARG L4T_TAG=r36.3.0
FROM nvcr.io/nvidia/l4t-base:${L4T_TAG}

# Non-interactive apt.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# App source goes here. Replace with your actual robot app bring-up.
WORKDIR /opt/bebop-app
COPY . /opt/bebop-app

# Expose anything the robot app needs; with network_mode=host this is
# informational only.
# EXPOSE 8080

# The robot app runs as PID 1 of the container. Use exec form so signals
# propagate correctly.
CMD ["python3", "-u", "main.py"]
