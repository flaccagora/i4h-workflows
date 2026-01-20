#!/bin/bash

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

set -e

# Define project root (assuming we are running inside the container at /workspace/i4h-workflows)
PROJECT_ROOT="/workspace/i4h-workflows"
echo "Project Root: $PROJECT_ROOT"

# Detect Python
# In the official Isaac Sim container, we should use the bundled python wrapper
# usually located at /isaac-sim/python.sh to ensure environment compatibility.
if [ -f "/isaac-sim/python.sh" ]; then
    echo "Found Isaac Sim python wrapper at /isaac-sim/python.sh"
    PYTHON="/isaac-sim/python.sh"
elif command -v python3 &> /dev/null; then
    echo "Using system python3"
    PYTHON=python3
else
    echo "Error: No python interpreter found."
    exit 1
fi
echo "Using Python: $PYTHON"

# 1. Install pip tools
echo "Installing pip tools..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install setuptools==75.8.0 toml==0.10.2

# 2. Install i4h-asset-catalog
echo "Installing i4h-asset-catalog..."
$PYTHON -m pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git@v0.3.0 \
    --extra-index-url https://pypi.nvidia.com

# 3. Install Isaac Lab
# NOTE: original script installs IsaacLab release/2.3.0
ISAACLAB_DIR="$PROJECT_ROOT/third_party/IsaacLab"
mkdir -p "$PROJECT_ROOT/third_party"

if [ -d "$ISAACLAB_DIR" ]; then
    echo "IsaacLab directory exists. Skipping clone."
else
    echo "Cloning IsaacLab (release/2.3.0)..."
    git clone -b release/2.3.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

# Ensure IsaacLab can find the Isaac Sim installation by creating a symlink to _isaac_sim
if [ -d "/isaac-sim" ]; then
    echo "Linking /isaac-sim to $ISAACLAB_DIR/_isaac_sim"
    # This ensures isaaclab.sh finds /isaac-sim/python.sh at _isaac_sim/python.sh
    ln -sf /isaac-sim "$ISAACLAB_DIR/_isaac_sim"
fi

echo "Installing IsaacLab..."
pushd "$ISAACLAB_DIR"
# We attempt to run the install script. 
# If this is an isaac-sim container, we rely on the environment being ready.
if [ -f "isaaclab.sh" ]; then
    # --install installs dependencies
    yes Yes | ./isaaclab.sh --install
else
    echo "isaaclab.sh not found, falling back to direct pip install..."
    $PYTHON -m pip install -e .
fi
popd

# 4. Patch IsaacLab (required as per original setup)
echo "Applying patches to IsaacLab..."
if [ -f "$ISAACLAB_DIR/source/isaaclab/isaaclab/utils/math.py" ]; then
    sed -i '/^[[:space:]]*import omni\.log[[:space:]]*$/d' "$ISAACLAB_DIR/source/isaaclab/isaaclab/utils/math.py"
    sed -i -E 's/^([[:space:]]*)omni\.log\.warn\(/\1import omni.log\
\1omni.log.warn(/g' "$ISAACLAB_DIR/source/isaaclab/isaaclab/utils/math.py"
else
    echo "Warning: math.py not found, skipping patch."
fi

# 5. Install robotic surgery extensions
EXTS_DIR="$PROJECT_ROOT/workflows/robotic_surgery/scripts/simulation/exts"
echo "Installing workflow extensions from $EXTS_DIR..."
if [ -d "$EXTS_DIR/robotic.surgery.assets" ]; then
    $PYTHON -m pip install --no-build-isolation -e $EXTS_DIR/robotic.surgery.assets
else
    echo "Warning: robotic.surgery.assets not found."
fi

if [ -d "$EXTS_DIR/robotic.surgery.tasks" ]; then
    $PYTHON -m pip install --no-build-isolation -e $EXTS_DIR/robotic.surgery.tasks
else
    echo "Warning: robotic.surgery.tasks not found."
fi

echo "Container dependency setup script finished."
