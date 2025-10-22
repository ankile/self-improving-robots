# SLURM Scripts and Setup

This directory contains all scripts and documentation for deploying the self-improving-robots training pipeline on SLURM-based clusters.

## Quick Start

```bash
# 1. Run one-time setup on cluster
bash slurm_setup.sh

# 2. Submit training job
bash slurm_submit_training.sh --repo-id my_dataset --job-name my_training

# 3. Monitor job
squeue -u $USER
tail -f ../logs/slurm-*.out
```

## Files Overview

### Installation & Setup

**`slurm_setup.sh`** (4.8 KB)
- Complete automated setup script
- Clones all required repositories
- Creates conda environment with Python 3.11
- Installs all dependencies
- Verifies installation
- **Run once per cluster**

Usage:
```bash
bash slurm_setup.sh [--base-dir BASE_DIR] [--env-name ENV_NAME]
```

### Job Submission

**`slurm_submit_training.sh`** (6.8 KB) - ⭐ **START HERE**
- Convenient wrapper for submitting training jobs
- Generates SLURM script with your configuration
- Supports dry-run mode to preview before submitting
- Handles all common parameters
- **Recommended for most users**

Usage:
```bash
bash slurm_submit_training.sh --repo-id DATASET_ID --job-name JOB_NAME [OPTIONS]
```

Examples:
```bash
# Basic
bash slurm_submit_training.sh --repo-id ankile/square-v1 --job-name square_v1

# With custom hyperparameters
bash slurm_submit_training.sh \
  --repo-id my_dataset \
  --job-name custom \
  --batch-size 32 \
  --training-steps 50000 \
  --lr 5e-05

# Preview without submitting
bash slurm_submit_training.sh --repo-id test --job-name preview --dry-run
```

**`slurm_train_act.sh`** (3.4 KB)
- Raw SLURM job script
- Direct sbatch submission
- More control over SLURM settings
- **For advanced users**

Usage:
```bash
sbatch slurm_train_act.sh --repo-id DATASET_ID [OPTIONS]
```

### Documentation

**`DEPLOYMENT_CHECKLIST.md`**
- Step-by-step checklist for new cluster deployments
- Pre-deployment verification
- Cluster assessment
- First job setup and verification
- Troubleshooting guide
- **Use for first-time cluster setup**

### Main Documentation Files (in parent directory)

**`../SLURM_SETUP.md`** (9.6 KB)
- Comprehensive setup guide
- Detailed explanation of each step
- GPU configuration
- Multi-GPU training
- Advanced usage patterns
- Troubleshooting
- **Read for detailed information**

**`../SLURM_QUICK_REFERENCE.md`** (4.9 KB)
- Quick command reference
- Common options and examples
- Job monitoring commands
- Hyperparameter sweep examples
- Useful one-liners
- **Bookmark for quick lookups**

## Installation Flow

```
First Time on Cluster
         |
         v
1. Clone Repository
   git clone https://github.com/ankile/self-improving-robots.git
   cd self-improving-robots
         |
         v
2. Run Setup Script (one-time)
   bash scripts/slurm_setup.sh
         |
         v
3. Submit Training Job
   bash scripts/slurm_submit_training.sh --repo-id MY_DATASET --job-name MY_JOB
         |
         v
4. Monitor Progress
   squeue -u $USER
   tail -f logs/slurm-*.out
         |
         v
✓ Training Complete!
```

## Common Tasks

### Submit a Training Job

```bash
bash slurm_submit_training.sh \
  --repo-id dataset_id \
  --job-name descriptive_name
```

### Monitor Running Jobs

```bash
# List all your jobs
squeue -u $USER

# Watch specific job
watch squeue -j JOB_ID

# View output
tail -f logs/slurm-JOB_ID.out
```

### Cancel a Job

```bash
scancel JOB_ID
```

### Test Configuration (Dry Run)

```bash
bash slurm_submit_training.sh \
  --repo-id test_dataset \
  --job-name test \
  --dry-run
```

### Run with Custom GPU Count

```bash
bash slurm_submit_training.sh \
  --repo-id my_dataset \
  --job-name multi_gpu \
  --gpus 2 \
  --cpus 16
```

### Submit Multiple Jobs (Hyperparameter Sweep)

```bash
for bs in 8 16 32; do
  bash slurm_submit_training.sh \
    --repo-id dataset \
    --job-name "sweep_bs${bs}" \
    --batch-size $bs
done
```

## Command Reference

### Setup Script
```bash
bash slurm_setup.sh [--base-dir DIR] [--env-name NAME]

Options:
  --base-dir DIR    Base directory for all repos (default: current)
  --env-name NAME   Conda environment name (default: sir)
```

### Job Submission Script
```bash
bash slurm_submit_training.sh --repo-id REPO_ID [OPTIONS]

Key Options:
  --repo-id REPO            Dataset repo ID (required)
  --job-name NAME           Job name (default: act-training)
  --batch-size N            Batch size (default: 16)
  --training-steps N        Training steps (default: 10000)
  --lr LR                   Learning rate (default: 1e-05)
  --gpus N                  GPU count (default: 1)
  --time HH:MM:SS           Time limit (default: 72:00:00)
  --no-wandb                Disable W&B logging
  --dry-run                 Preview without submitting

See ../SLURM_SETUP.md for full option list
```

## Troubleshooting

### Setup fails
- Check conda/mamba is installed: `conda --version`
- Check git is available: `git --version`
- Check disk space: `df -h`
- See `DEPLOYMENT_CHECKLIST.md` for detailed troubleshooting

### Job won't start
- Check partition exists: `sinfo`
- Check quota: `scontrol show job JOB_ID`
- Try with `--time 1:00:00` first (shorter time)

### Job runs out of memory
- Reduce batch size: `--batch-size 8`
- Reduce training steps: `--training-steps 1000`
- Check node capacity: `sinfo -N`

### Slow training
- Increase batch size: `--batch-size 32`
- Use more CPUs: `--cpus 16`
- Check node load: `htop` or `sinfo`

## Getting Help

1. **First time?** → Read `DEPLOYMENT_CHECKLIST.md`
2. **Need quick commands?** → Check `../SLURM_QUICK_REFERENCE.md`
3. **Detailed info?** → See `../SLURM_SETUP.md`
4. **Job failed?** → View logs: `cat logs/slurm-*.err`
5. **Script help?** → Run script with no args for usage

## Best Practices

✓ **DO:**
- Test configuration with `--dry-run` first
- Test locally before submitting big jobs
- Monitor W&B for real-time metrics
- Use descriptive job names
- Check `sinfo` before submitting

✗ **DON'T:**
- Submit huge training runs without testing
- Set unreasonably long time limits
- Ignore job errors
- Forget to activate environment
- Use partition without verification

## File Structure

```
scripts/
├── slurm_setup.sh              ⭐ Run once for setup
├── slurm_submit_training.sh    ⭐ Use for job submission
├── slurm_train_act.sh          For advanced users
├── README.md                   This file
├── DEPLOYMENT_CHECKLIST.md     First-time setup guide
├── SLURM_SETUP.md             Detailed documentation
└── SLURM_QUICK_REFERENCE.md   Command cheat sheet
```

## Next Steps

1. **Setup**: Run `bash slurm_setup.sh`
2. **Test**: Submit a test job with `--dry-run`
3. **Monitor**: Use `squeue` and check logs
4. **Scale**: Run production training jobs
5. **Sweep**: Run hyperparameter sweeps
6. **Analyze**: Check W&B dashboard for results

## Version Info

- Created: October 2024
- Python: 3.11+
- Compatible with: Any SLURM cluster
- Dependencies: conda/mamba, git

## Support

For issues:
1. Check logs: `cat logs/slurm-*.err`
2. Review `DEPLOYMENT_CHECKLIST.md`
3. Consult `../SLURM_SETUP.md`
4. Check SLURM documentation: `man sbatch`

---

**Pro Tip**: Bookmark `../SLURM_QUICK_REFERENCE.md` for frequently used commands!
