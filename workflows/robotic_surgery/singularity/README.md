# Robotic Surgery Singularity Container for HPC Clusters

This guide provides instructions for running robotic surgery simulations using Singularity containers on HPC clusters. Singularity is the preferred container runtime for most HPC environments due to better integration with shared filesystems and resource schedulers.

## Prerequisites

- **Singularity/Apptainer** (v3.8+) installed on the HPC cluster
- **NVIDIA GPU support** with CUDA-capable GPUs
- Access to the HPC cluster with appropriate resource allocation
- A Docker image of the robotic_surgery container already built and available

## Building the Singularity Image

### Option 1: Convert from Docker Image (Recommended for HPC)

If you have built the Docker image locally, push it to a container registry (e.g., Docker Hub, GitLab Registry):

```sh
# Tag and push Docker image
docker tag robotic_surgery:latest <registry>/robotic_surgery:latest
docker push <registry>/robotic_surgery:latest
```

On the HPC cluster, pull and convert to Singularity:

```sh
# Pull Docker image and convert to Singularity
singularity pull robotic_surgery.sif docker://<registry>/robotic_surgery:latest

# Or if using Docker Hub
singularity pull robotic_surgery.sif docker://docker.io/<username>/robotic_surgery:latest
```

### Option 2: Build Directly on HPC (if permitted)

Create a `robotic_surgery.def` recipe file:

```singularity
Bootstrap: docker
From: <registry>/robotic_surgery:latest

%post
    # Additional HPC-specific configurations if needed
    apt-get update
    apt-get install -y openssh-client

%environment
    export OMNI_KIT_ACCEPT_EULA=Y
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y

%runscript
    exec /bin/bash "$@"
```

Build with:

```sh
singularity build robotic_surgery.sif robotic_surgery.def
```

## Running on HPC Clusters

### Basic Headless Execution (No GUI)

For most HPC environments, run in headless mode:

```sh
# Simple execution
singularity exec --nv robotic_surgery.sif python -c "import holoscan; print(holoscan.__version__)"

# With volume binds for data
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    -B /home/$USER/.cache/i4h-assets:/root/.cache/i4h-assets \
    robotic_surgery.sif \
    bash -c "cd /workspace && python your_script.py"
```

### Running Simulation Scripts

```sh
# Activate conda environment and run simulation
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    robotic_surgery.sif \
    conda run -n robotic_surgery \
    python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py
```

### Headless WebRTC Streaming Mode

For remote visualization over network:

```sh
# Run with WebRTC streaming on HPC
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    robotic_surgery.sif \
    conda run -n robotic_surgery \
    python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py --livestream 2
```

## Submitting Jobs via Cluster Scheduler

### SLURM Job Script Example

Create `submit_robotic_surgery.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=robotic_surgery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Load required modules
module load cuda/11.8
module load singularity

# Set paths
export SINGULARITY_IMAGE=/path/to/robotic_surgery.sif
export WORKSPACE=/scratch/$USER/robotic_surgery_run

# Create necessary directories
mkdir -p $WORKSPACE/logs
mkdir -p $WORKSPACE/outputs

# Run simulation
singularity exec --nv \
    -B $WORKSPACE:/workspace \
    -B /home/$USER/.cache/i4h-assets:/root/.cache/i4h-assets \
    $SINGULARITY_IMAGE \
    conda run -n robotic_surgery \
    python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py

echo "Job completed at $(date)"
```

Submit with:

```sh
sbatch submit_robotic_surgery.sbatch
```

### PBS/Torque Job Script Example

Create `submit_robotic_surgery.pbs`:

```bash
#!/bin/bash
#PBS -N robotic_surgery
#PBS -l select=1:ngpus=1:mem=16gb
#PBS -l walltime=01:00:00
#PBS -o logs/job_${PBS_JOBID}.out
#PBS -e logs/job_${PBS_JOBID}.err

module load cuda/11.8
module load singularity

SINGULARITY_IMAGE=/path/to/robotic_surgery.sif
WORKSPACE=$TMPDIR/robotic_surgery_run

mkdir -p $WORKSPACE/logs
mkdir -p $WORKSPACE/outputs

singularity exec --nv \
    -B $WORKSPACE:/workspace \
    -B $HOME/.cache/i4h-assets:/root/.cache/i4h-assets \
    $SINGULARITY_IMAGE \
    conda run -n robotic_surgery \
    python workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/reach_psm_sm.py

echo "Job completed at $(date)"
```

Submit with:

```sh
qsub submit_robotic_surgery.pbs
```

## Volume Binding and Data Management

### Common Binding Patterns

| Purpose | Docker | Singularity |
|---------|--------|-------------|
| Working directory | `-v ~/work:/workspace` | `-B ~/work:/workspace` |
| Output data | `-v ~/outputs:/outputs` | `-B ~/outputs:/outputs` |
| Cache files | `-v ~/.cache:/root/.cache` | `-B ~/.cache:/root/.cache` |
| Scratch space | `-v /tmp:/tmp` | `-B /tmp:/tmp` (automatic) |

### HPC-Specific Bindings

```sh
# Bind scratch space (faster I/O)
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    robotic_surgery.sif \
    bash

# Bind home directory (for accessing data and scripts)
singularity exec --nv \
    -B $HOME:/home_mount \
    robotic_surgery.sif \
    bash

# Multiple bindings
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    -B $HOME/.cache/i4h-assets:/root/.cache/i4h-assets \
    -B $HOME/projects:/projects \
    robotic_surgery.sif \
    bash
```

## GPU Configuration

### NVIDIA GPU Support

Singularity automatically detects NVIDIA GPUs with the `--nv` flag:

```sh
# Enable NVIDIA GPU access
singularity exec --nv robotic_surgery.sif nvidia-smi

# Specify specific GPUs (optional)
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
singularity exec --nv robotic_surgery.sif python script.py
```

### Checking GPU Access

```sh
singularity exec --nv robotic_surgery.sif python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
"
```

## Environment Variables

Set environment variables inside Singularity containers:

```sh
# Via command line
singularity exec \
    --env "OMNI_KIT_ACCEPT_EULA=Y" \
    --env "ACCEPT_EULA=Y" \
    robotic_surgery.sif \
    python script.py

# Via environment file
export SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=Y
export SINGULARITYENV_ACCEPT_EULA=Y
singularity exec --nv robotic_surgery.sif python script.py
```

## Interactive Session on HPC

### Login Node (Development/Testing)

```sh
# Start interactive shell
singularity shell --nv robotic_surgery.sif

# Inside container
> conda activate robotic_surgery
> python -c "import holoscan; print('Holoscan ready')"
> exit
```

### Compute Node (via salloc/qsub)

```sh
# SLURM: Allocate resources and run interactively
salloc --gres=gpu:1 --mem=16G --time=1:00:00
singularity shell --nv robotic_surgery.sif

# PBS: Similar approach
qsub -I -l select=1:ngpus=1:mem=16gb
singularity shell --nv robotic_surgery.sif
```

## Troubleshooting

### Common Issues

**GPU Not Detected**
```sh
# Verify --nv flag is used
singularity exec --nv robotic_surgery.sif nvidia-smi

# Check CUDA modules are loaded
module list
module load cuda

# Check Singularity version
singularity --version  # Should be 3.8+
```

**Permission Denied Errors**
```sh
# Ensure readable paths for bindings
ls -la /path/to/bind

# Use absolute paths for all bindings
singularity exec -B /absolute/path robotic_surgery.sif bash
```

**Out of Memory**
```sh
# Request more memory in job script
#SLURM --mem=32G

# Check memory usage
singularity exec --nv robotic_surgery.sif free -h
```

**Mount Path Not Found**
```sh
# Create the binding path if it doesn't exist
mkdir -p /path/to/bind

# Verify paths before running
singularity exec robotic_surgery.sif ls /mounted/path
```

### Debugging

Enable verbose output:

```sh
# SINGULARITY_DEBUG for more information
SINGULARITY_DEBUG=1 singularity exec --nv robotic_surgery.sif bash

# Check container contents
singularity inspect robotic_surgery.sif

# Run diagnostics
singularity exec --nv robotic_surgery.sif python -c "
import sys
print('Python:', sys.version)
try:
    import holoscan
    print('Holoscan:', holoscan.__version__)
except ImportError:
    print('Holoscan not available')
"
```

## Performance Considerations

### Cache Directory Management

For multiple jobs, cache directories can grow large. Configure them in job scripts:

```bash
# Create per-job cache directories
CACHE_DIR=/scratch/$USER/job_${SLURM_JOB_ID}/cache
mkdir -p $CACHE_DIR

singularity exec --nv \
    -B $CACHE_DIR:/root/.cache \
    robotic_surgery.sif \
    python script.py

# Clean up after job
rm -rf $CACHE_DIR
```

### I/O Optimization

For large data transfers:

```sh
# Use local scratch for faster I/O
singularity exec --nv \
    -B /scratch/$USER:/workspace \
    robotic_surgery.sif \
    bash -c "cp /workspace/input/* /tmp/ && python process.py && cp /tmp/output/* /workspace/"
```

## Additional Resources

- [Singularity Documentation](https://sylabs.io/docs/)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [HPC Best Practices](https://docs.sylabs.io/all-user/latest/user-guide/cli/singularity_exec.html)

## Getting Help

For issues with:
- **Singularity**: Check cluster documentation or contact HPC support
- **Isaac Sim**: Refer to the [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- **Robotic Surgery Workflows**: See the main [README.md](../README.md)
