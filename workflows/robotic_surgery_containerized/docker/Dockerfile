# syntax=docker/dockerfile:1.4

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Users can override this with --build-arg ISAAC_SIM_IMAGE=...
# Ensure the tag matches the requirements (e.g. Isaac Sim 4.2+, 5.0 if available)
ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:4.2.0
FROM ${ISAAC_SIM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Install system dependencies (mirrored from original Dockerfile)
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        jq \
        vim \
        git \
        xvfb \
        build-essential \
        cmake \
        vulkan-tools \
        unzip \
        lsb-release \
        libglib2.0-0 \
        libdbus-1-3 \
        libopengl0 \
        libxcb-keysyms1 \
        libglu1-mesa && \
    rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace/i4h-workflows

# Copy the repository
COPY . /workspace/i4h-workflows

# Run the env setup script adapted for container usage
# This script installs IsaacLab and workflow extensions without conda
RUN bash workflows/robotic_surgery_containerized/scripts/install_deps.sh

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display

# Default entrypoint
CMD ["/bin/bash"]
