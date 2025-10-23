# SLURM Cluster Setup Guide

This guide explains how to set up and run the self-improving-robots training pipeline on a SLURM-based cluster.

## Quick Start

```bash
# 1. Clone or pull the repository
git clone https://github.com/ankile/self-improving-robots.git
cd self-improving-robots

# 2. Run the setup script
bash scripts/slurm_setup.sh

# 3. Submit a training job
bash scripts/slurm_submit_training.sh --repo-id my_dataset --job-name my_training
```

## Step-by-Step Setup

### 1. Initial Setup Script

The `scripts/slurm_setup.sh` script automates the entire setup process:

```bash
bash scripts/slurm_setup.sh [--base-dir BASE_DIR] [--env-name ENV_NAME]
```

**What it does:**

- Clones all required repositories (self-improving-robots, lerobot, robosuite, maniskill)
- Creates a conda environment with Python 3.11
- Installs PyTorch (CPU or GPU - adjust as needed)
- Installs all dependencies in editable mode
- Verifies the installation

**Prerequisites:**

- `micromamba`, `mamba`, or `conda` must be installed (micromamba preferred for SLURM)
  - On SLURM: Try `module load micromamba` or `module load miniconda3`
  - Quick install: `"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`
  - Or see: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- Git must be available
- 50+ GB of free disk space (for repos and dependencies)

**Example:**

```bash
# Setup in current directory with environment name "sir"
bash scripts/slurm_setup.sh

# Setup in a specific directory
bash scripts/slurm_setup.sh --base-dir /path/to/projects --env-name research-env
```

### 2. Manual Setup (if needed)

If the script doesn't work, you can manually set up:

```bash
# Create conda environment
conda create -n sir python=3.11
conda activate sir

# Clone repositories
git clone https://github.com/lerobotics/lerobot.git
git clone https://github.com/ARISE-Initiative/robosuite.git
git clone https://github.com/haosulab/ManiSkill.git maniskill
git clone https://github.com/ankile/self-improving-robots.git

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
cd lerobot && pip install -e . && cd ..
cd robosuite && pip install -e . && cd ..
cd maniskill && pip install -e . && cd ..
cd self-improving-robots && pip install -e . && cd ..

# Install training dependencies
pip install wandb hydra-core tensorboard tqdm pyyaml
```

## Submitting Training Jobs

### Method 1: Convenient Wrapper (Recommended)

Use the `slurm_submit_training.sh` wrapper for easy job submission:

```bash
bash scripts/slurm_submit_training.sh \
  --repo-id my_dataset \
  --job-name my_training
```

**Common options:**

```bash
--repo-id REPO_ID              # HuggingFace dataset ID (required)
--job-name NAME                # Job name (default: act-training)
--root DIR                     # Local data directory (default: ./data)
--batch-size N                 # Batch size (default: 16)
--lr LR                        # Learning rate (default: 1e-05)
--training-steps N             # Number of steps (default: 10000)
--eval-freq N                  # Evaluation frequency (default: 1000)
--eval-episodes N              # Episodes per eval (default: 5)
--chunk-size N                 # Action chunk size (default: 20)
--partition NAME               # SLURM partition (default: gpu)
--gpus N                       # Number of GPUs (default: 1)
--cpus N                       # Number of CPUs (default: 8)
--time HH:MM:SS                # Time limit (default: 72:00:00)
--mem GB                       # Memory (default: 64GB)
--no-wandb                     # Disable Weights & Biases logging
--no-video                     # Disable video saving
--wandb-project NAME           # W&B project name
--env-name NAME                # Conda environment name (default: sir)
--dry-run                      # Show command without submitting
```

**Examples:**

```bash
# Basic training with minimal config
bash scripts/slurm_submit_training.sh \
  --repo-id ankile/square-v1 \
  --job-name square_training

# Training with custom hyperparameters
bash scripts/slurm_submit_training.sh \
  --repo-id ankile/lift-v1 \
  --job-name lift_32bs \
  --batch-size 32 \
  --training-steps 50000 \
  --lr 5e-05

# Long training with 2 GPUs and extended time
bash scripts/slurm_submit_training.sh \
  --repo-id large_dataset \
  --job-name long_training \
  --gpus 2 \
  --time 120:00:00 \
  --training-steps 100000

# Preview command without submitting
bash scripts/slurm_submit_training.sh \
  --repo-id test_dataset \
  --job-name preview \
  --dry-run
```

### Method 2: Direct SLURM Submission

For more control, you can directly submit the SLURM script:

```bash
sbatch scripts/slurm_train_act.sh \
  --repo-id ankile/square-v1 \
  --batch-size 32 \
  --training-steps 50000
```

Edit `scripts/slurm_train_act.sh` to customize SLURM headers (partition, time, memory, etc.).

### Method 3: Create Custom Job Script

Create a custom SLURM script for your specific setup:

```bash
#!/bin/bash
#SBATCH --job-name=my-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=64GB

eval "$(conda shell.bash hook)"
conda activate sir

python -m sir.training.train_act \
  --repo-id my_dataset \
  --root ./data \
  --batch-size 16 \
  --training-steps 10000 \
  --use-wandb \
  --save-video
```

## Monitoring Jobs

### Check job status

```bash
# List all your jobs
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# Watch job in real-time
watch squeue -j JOB_ID
```

### View output

```bash
# View current output
tail -f logs/slurm-JOB_ID.out

# View entire output
cat logs/slurm-JOB_ID.out

# View errors
cat logs/slurm-JOB_ID.err
```

### Cancel job

```bash
scancel JOB_ID
```

## GPU Configuration

The default setup assumes a single GPU. For multi-GPU training:

1. Update SLURM settings:

```bash
bash scripts/slurm_submit_training.sh \
  --repo-id my_dataset \
  --gpus 2 \
  --cpus 16
```

2. Or manually in SLURM script:

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
```

## PyTorch Version for GPU

If using GPU nodes, install GPU-enabled PyTorch:

```bash
# In setup script, replace the PyTorch installation line with:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 11.8 with PyTorch. Adjust the CUDA version based on your cluster:

- `cu118` for CUDA 11.8
- `cu121` for CUDA 12.1
- `cu124` for CUDA 12.4

Check your cluster's CUDA version with `nvcc --version`.

## Troubleshooting

### Setup issues

**Issue: "conda: command not found"**

- Solution: Install conda or mamba first
- Mamba: https://github.com/conda-forge/miniforge

**Issue: "No space left on device"**

- Solution: Check disk space with `df -h`, or use a different base directory
- Repos require ~20GB, dependencies another ~20GB

**Issue: "Permission denied"**

- Solution: Make scripts executable: `chmod +x scripts/*.sh`

### Runtime issues

**Issue: "CUDA out of memory"**

- Solution: Reduce batch size or use fewer GPUs

```bash
bash scripts/slurm_submit_training.sh \
  --repo-id my_dataset \
  --batch-size 8  # Reduced from 16
```

**Issue: "Dataset not found"**

- Ensure HuggingFace Hub dataset exists and is public or you have access
- Check repo-id is correct: `--repo-id username/dataset-name`

**Issue: "W&B authentication failed"**

- Solution: Login to W&B first:

```bash
wandb login
```

- Or use `--no-wandb` flag

**Issue: Job stuck or not starting**

- Check partition availability: `sinfo`
- Check node status: `sinfo -N`
- Reduce resource requirements and try again

## Data Management

### Download datasets locally

Datasets are automatically downloaded on first use. To pre-download:

```bash
python -c "
from lerobot.datasets import LeRobotDataset
dataset = LeRobotDataset('ankile/square-v1')
"
```

### Organize data across jobs

Store common data in a shared location:

```bash
# Download to shared location
export DATA_DIR=/shared/lerobot_data
bash scripts/slurm_submit_training.sh \
  --repo-id my_dataset \
  --root $DATA_DIR
```

### Use multiple datasets in single job

Modify training script or create symbolic links:

```bash
ln -s /path/to/dataset1 ./data/dataset1
ln -s /path/to/dataset2 ./data/dataset2
```

## Advanced Usage

### Job arrays for hyperparameter sweeps

Create multiple jobs with different parameters:

```bash
for batch_size in 8 16 32; do
  for lr in 1e-5 5e-5 1e-4; do
    bash scripts/slurm_submit_training.sh \
      --repo-id my_dataset \
      --job-name "sweep_bs${batch_size}_lr${lr}" \
      --batch-size $batch_size \
      --lr $lr
  done
done
```

### Chain multiple jobs

Use job dependencies to run jobs sequentially:

```bash
# Submit preprocessing job
JOB1=$(sbatch script1.sh | awk '{print $NF}')

# Submit training job that depends on preprocessing
sbatch --dependency=afterok:$JOB1 scripts/slurm_train_act.sh
```

### Environment variables

Export custom settings before submitting:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
bash scripts/slurm_submit_training.sh --repo-id my_dataset
```

## Best Practices

1. **Test locally first**: Run a short training locally before submitting to SLURM

```bash
python -m sir.training.train_act \
  --repo-id my_dataset \
  --training-steps 100  # Short test run
```

2. **Use reasonable time limits**: Start conservative, increase as needed

```bash
--time 24:00:00  # 24 hours for initial testing
```

3. **Monitor W&B**: Track training progress in real-time at wandb.ai

4. **Save checkpoints**: Enable video and checkpoint saving

```bash
--save-video --eval-freq 500
```

5. **Use dry-run first**: Preview your job before submitting

```bash
--dry-run
```

## Support

For issues with:

- **Setup**: Check error messages in terminal
- **SLURM**: Contact your cluster administrator
- **Training**: Check logs in `logs/slurm-*.out`
- **Code**: See project README and GitHub issues

## Further Reading

- [SLURM Documentation](https://slurm.schedmd.com/)
- [LeRobot Documentation](https://github.com/lerobotics/lerobot)
- [Weights & Biases Integration](https://docs.wandb.ai/)
