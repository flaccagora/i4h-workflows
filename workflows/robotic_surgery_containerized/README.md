# Robotic Surgery Workflow - Containerized Build

This directory contains resources to build the Robotic Surgery Workflow directly on top of an NVIDIA Isaac Sim docker container, bypassing the Conda environment setup.

## Overview

Instead of installing Isaac Sim via pip into a conda environment (which requires complex dependency management), this approach builds a Docker image extending the official `nvcr.io/nvidia/isaac-sim` image.

## Prerequisites

- **NVIDIA GPU** with supported drivers.
- **Docker** with **NVIDIA Container Toolkit** configured.

## Directory Structure

- `docker/`: Contains the `Dockerfile` for the containerized build.
- `scripts/`: Contains `install_deps.sh` which sets up IsaacLab and workflow extensions inside the container.

## Usage

### 1. Build the Container

From the root of the `i4h-workflows` repository:

```bash
# Build the image (defaulting to nvcr.io/nvidia/isaac-sim:4.2.0)
docker build \
  -t i4h-robotic-surgery:latest \
  -f workflows/robotic_surgery_containerized/docker/Dockerfile \
  .
```

**Note**: You can specify a different base image using `--build-arg`:
```bash
docker build \
  --build-arg ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:4.0.0 \
  -t i4h-robotic-surgery:latest \
  -f workflows/robotic_surgery_containerized/docker/Dockerfile \
  .
```

### 2. Run the Container

Run the container with NVIDIA runtime enabled:

```bash
docker run --name i4h-sim --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm \
  -v $(pwd):/workspace/i4h-workflows \
  i4h-robotic-surgery:latest
```

Inside the container, you can run the workflow scripts (ensure you use the python environment where dependencies were installed, typically default `python` or `python3`).

### 3. Verify Installation

Inside the container:

```bash
python -c "import isaaclab; print('Isaac Lab installed successfully')"
python -c "import robotic.surgery.tasks; print('Robotic Surgery Tasks installed successfully')"
```
