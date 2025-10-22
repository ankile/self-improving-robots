# SLURM Deployment Checklist

Complete this checklist before deploying to a new SLURM cluster.

## Pre-Deployment (Local Testing)

- [ ] Code runs locally without errors
- [ ] Dataset exists on HuggingFace Hub or is accessible
- [ ] W&B account is set up (if using logging)
- [ ] GPU tests pass if using GPU training
- [ ] All dependencies install without conflicts

Quick local test:
```bash
python -m sir.training.train_act --repo-id test_dataset --training-steps 100
```

## Cluster Assessment

- [ ] Cluster has SLURM available (`sbatch --version` works)
- [ ] Have an account with proper quota
- [ ] Know the available partitions (`sinfo`)
- [ ] Know GPU types available (`sinfo -N`)
- [ ] Know the project/account code (if required)
- [ ] Storage quota is sufficient (~50GB minimum)
- [ ] Network access to HuggingFace Hub is available
- [ ] Can access home directory and any shared storage

Check cluster info:
```bash
sinfo
sinfo -N
scontrol show config | grep -E "SlurmdTimeout|PrologTimeout"
```

## Initial Setup on Cluster

- [ ] SSH access to cluster works
- [ ] Conda/mamba is available
```bash
which conda
# or
which mamba
```

- [ ] Git is available
```bash
git --version
```

- [ ] Have 50+ GB free storage
```bash
df -h
```

- [ ] Run setup script
```bash
bash scripts/slurm_setup.sh
```

- [ ] Setup completes without errors
- [ ] Installation verification passes
```bash
python -c "import sir.training.train_act; print('✓ OK')"
```

## SLURM Configuration Review

### For CPU-Only Training

In `slurm_setup.sh`:
```bash
# ✓ PyTorch CPU version is installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### For GPU Training

Before running setup:
1. Check available CUDA versions:
```bash
nvidia-smi  # Shows CUDA version
nvcc --version
```

2. Modify PyTorch installation line in `slurm_setup.sh`:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Update `slurm_train_act.sh` SLURM headers:
```bash
#SBATCH --partition=gpu        # Adjust based on your cluster
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
```

## First Training Job

### Option 1: Preview First (Recommended)

```bash
bash scripts/slurm_submit_training.sh \
  --repo-id test_dataset \
  --job-name first_test \
  --training-steps 100 \
  --batch-size 8 \
  --dry-run
```

Review the generated job script and check:
- [ ] Paths are correct
- [ ] Module names are correct
- [ ] Arguments are as expected

### Option 2: Submit Small Test Job

```bash
bash scripts/slurm_submit_training.sh \
  --repo-id ankile/square-v1 \
  --job-name test_job \
  --training-steps 100 \
  --partition gpu \
  --time 1:00:00
```

Monitor the job:
```bash
squeue -u $USER
tail -f logs/slurm-*.out
```

Expected output:
```
✓ Initialized Weights & Biases
✓ Dataset metadata loaded
✓ Dataset loaded
✓ Evaluation environment created
Starting training...
step: 0/100 loss: ...
```

## After First Job

### If Job Failed

- [ ] Check output logs: `cat logs/slurm-*.err`
- [ ] Check error types
- [ ] Try one of these:
  - Reduce batch size
  - Reduce training steps
  - Increase time limit
  - Adjust partition
  - Verify dataset exists

### If Job Succeeded

- [ ] Check output quality
- [ ] Verify checkpoints saved
- [ ] Check W&B dashboard (if using)
- [ ] Review hyperparameters for scaling

## Production Deployment

### Resource Allocation

- [ ] Determine training duration (test with 1000 steps)
- [ ] Set appropriate time limits (1.5x estimated)
- [ ] Request appropriate GPU count
- [ ] Request appropriate memory

Estimate training time:
```bash
# Start with short run to measure time per step
--training-steps 1000 --eval-freq 1000
# Monitor: tail -f logs/slurm-*.out
# Calculate: time_per_step * total_steps
```

### Job Configuration

- [ ] Appropriate `--partition` for your work
- [ ] Appropriate `--time` (not too long or too short)
- [ ] Appropriate `--gpus` (1 for most cases)
- [ ] Appropriate batch size (8-64 depending on GPU)

### Data Management

- [ ] Datasets cache correctly
- [ ] Checkpoints save properly
- [ ] Video outputs work (if enabled)
- [ ] Logs are being collected

Check after first production run:
```bash
ls -lah checkpoints/
ls -lah logs/
```

### Monitoring

- [ ] Set up W&B alerts (if using)
- [ ] Know how to check job status (`squeue`)
- [ ] Know how to view logs (`tail -f`)
- [ ] Know how to cancel jobs (`scancel`)

### Scaling

- [ ] Test with different batch sizes
- [ ] Test with different learning rates
- [ ] Test with different training step counts
- [ ] Document successful configurations

## Troubleshooting Guide

### Common Issues

**Issue: Job won't start**
```bash
# Check SLURM status
sinfo
squeue -j JOB_ID

# Likely causes:
# 1. Partition doesn't exist
# 2. GPU quota exceeded
# 3. Time limit too short
```

**Issue: Out of memory**
```bash
# Reduce batch size
--batch-size 8  # from 16

# Or reduce image resolution in environment
```

**Issue: Slow training**
```bash
# Increase batch size
--batch-size 32

# Use all available CPUs
#SBATCH --cpus-per-task=16
```

**Issue: Network errors downloading data**
```bash
# Pre-download data:
python -c "from lerobot.datasets import LeRobotDataset; LeRobotDataset('dataset_id')"

# Then use --root flag to point to local cache
```

## Cluster-Specific Notes

Document your cluster configuration:

```
Cluster Name: _______________
SLURM Version: _______________
CUDA Version: _______________
Available Partitions: _______________
GPU Types: _______________
Max Time Limit: _______________
Storage Quota: _______________
Contact: _______________
```

## Verification Commands

Run these regularly to verify setup:

```bash
# Check environment
conda activate sir
python -m sir.training.train_act --help

# Check SLURM
sinfo
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State

# Check storage
df -h
du -sh checkpoints logs data

# Check GPU (if available)
nvidia-smi

# Check recent jobs
sacct -u $USER -S now-7days
```

## Documentation to Keep

- [ ] SLURM_SETUP.md - Complete setup guide
- [ ] SLURM_QUICK_REFERENCE.md - Quick commands
- [ ] This checklist - For next deployments
- [ ] Cluster documentation - For reference
- [ ] Job configurations - For reproducibility

## Final Sign-Off

- [ ] All checks passed
- [ ] First training job completed successfully
- [ ] Team has access to cluster and scripts
- [ ] Documentation is accessible
- [ ] Backup of configuration files made

Deployed by: _______________
Date: _______________
Cluster: _______________
Notes: _______________
