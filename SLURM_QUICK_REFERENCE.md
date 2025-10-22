# SLURM Setup - Quick Reference

## One-Time Setup

```bash
# Clone repo and setup everything
git clone https://github.com/ankile/self-improving-robots.git
cd self-improving-robots
bash scripts/slurm_setup.sh
```

## Submit Training Job

```bash
# Basic usage
bash scripts/slurm_submit_training.sh --repo-id DATASET_ID --job-name JOB_NAME

# Examples
bash scripts/slurm_submit_training.sh --repo-id ankile/square-v1 --job-name square_v1
bash scripts/slurm_submit_training.sh --repo-id ankile/lift-v1 --job-name lift_demo --batch-size 32
bash scripts/slurm_submit_training.sh --repo-id my_data --job-name test --dry-run  # Preview
```

## Monitor Jobs

```bash
# List all jobs
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# Watch output in real-time
tail -f logs/slurm-JOB_ID.out

# Get job info
scontrol show job JOB_ID

# Cancel job
scancel JOB_ID
```

## Common Options

```bash
--repo-id REPO              # HuggingFace dataset (required)
--job-name NAME             # Job name
--batch-size N              # Batch size (default: 16)
--training-steps N          # Training steps (default: 10000)
--lr LR                     # Learning rate (default: 1e-05)
--gpus N                    # Number of GPUs (default: 1)
--time HH:MM:SS             # Time limit (default: 72:00:00)
--no-wandb                  # Disable W&B logging
--dry-run                   # Preview job without submitting
```

## Full Setup with Different Options

```bash
# GPU-accelerated training
bash scripts/slurm_setup.sh --env-name gpu-training

# In a custom directory
bash scripts/slurm_setup.sh --base-dir /large_disk/projects --env-name research

# Manual setup (for debugging)
conda create -n sir python=3.11 -y
conda activate sir
pip install -e lerobot
pip install -e self-improving-robots
pip install wandb hydra-core
```

## Real-World Examples

### Training on a new dataset
```bash
bash scripts/slurm_submit_training.sh \
  --repo-id my_org/my_new_dataset \
  --job-name my_new_dataset_v1 \
  --batch-size 32 \
  --training-steps 50000
```

### Long training run with extended time
```bash
bash scripts/slurm_submit_training.sh \
  --repo-id large_dataset \
  --job-name long_training \
  --training-steps 100000 \
  --time 120:00:00 \
  --gpus 2
```

### Testing configuration before full training
```bash
bash scripts/slurm_submit_training.sh \
  --repo-id test_dataset \
  --job-name test_config \
  --training-steps 100 \
  --eval-freq 50 \
  --dry-run  # Review before submitting
```

### Training without W&B (offline)
```bash
bash scripts/slurm_submit_training.sh \
  --repo-id my_dataset \
  --job-name offline_training \
  --no-wandb
```

## Hyperparameter Sweep

```bash
# Submit multiple training jobs with different parameters
for bs in 16 32 64; do
  bash scripts/slurm_submit_training.sh \
    --repo-id my_dataset \
    --job-name "sweep_bs${bs}" \
    --batch-size $bs
done
```

## Job Status

```bash
# Show queued jobs
squeue -u $USER -t PENDING

# Show running jobs
squeue -u $USER -t RUNNING

# Show completed jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode --endtime=now-1day

# Show all jobs from today
sacct -u $USER --starttime=today
```

## Troubleshooting

```bash
# Check if environment exists
conda env list | grep sir

# Activate environment manually
conda activate sir

# Verify installation
python -m sir.training.train_act --help

# Check SLURM partition availability
sinfo

# Check node status
sinfo -N

# View setup logs (if setup failed)
cat setup.log

# Check GPU availability
nvidia-smi

# Test training locally first
python -m sir.training.train_act \
  --repo-id test_dataset \
  --training-steps 10  # Just 10 steps to test
```

## Important Paths

```bash
# Logs
logs/slurm-*.out      # Job output
logs/slurm-*.err      # Job errors

# Checkpoints
checkpoints/          # Saved models

# Data
data/                 # Datasets

# Scripts
scripts/slurm_setup.sh              # Initial setup
scripts/slurm_submit_training.sh    # Job submission wrapper
scripts/slurm_train_act.sh          # Raw SLURM script
```

## Environment Variables (if needed)

```bash
# Set before submitting job
export CUDA_VISIBLE_DEVICES=0,1    # GPU devices to use
export OMP_NUM_THREADS=8           # OpenMP threads
export WANDB_PROJECT=my_project    # W&B project
export WANDB_ENTITY=my_entity      # W&B entity
```

## Tips & Tricks

1. **Always use `--dry-run` first** to verify your job configuration
2. **Test locally** before submitting large training runs
3. **Monitor W&B** for real-time training metrics at wandb.ai
4. **Save outputs** for long runs: `--save-video --eval-freq 500`
5. **Start with conservative time limits**, increase as needed
6. **Use reasonable batch sizes**: 8-32 for most hardware
7. **Check cluster load** before submitting: `sinfo`

## Get Help

```bash
# Show all available options
bash scripts/slurm_submit_training.sh -h

# Check SLURM documentation
man sbatch
man squeue
man scancel

# View job details
scontrol show job JOB_ID
```
