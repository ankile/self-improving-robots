#!/bin/bash
#
# SLURM Job Submission Script for ACT Policy Training
#
# This script submits a training job to a SLURM cluster.
# The actual training is run in slurm_train_act_job.sh
#
# Usage:
#   sbatch scripts/slurm_train_act.sh --repo-id REPO_ID [--other-flags]
#
# Or use the wrapper:
#   bash scripts/slurm_submit_training.sh --repo-id REPO_ID --job-name my_job
#

# SLURM Configuration
#SBATCH --job-name=act-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Parse command line arguments
REPO_ID=""
ROOT="./data"
ENV_NAME="sir"
BATCH_SIZE="16"
LR="1e-05"
TRAINING_STEPS="10000"
EVAL_FREQ="1000"
EVAL_EPISODES="5"
CHUNK_SIZE="20"
SAVE_VIDEO="true"
USE_WANDB="true"
WANDB_PROJECT="act-training"

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --training-steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --eval-freq)
            EVAL_FREQ="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$REPO_ID" ]; then
    echo "Error: --repo-id is required"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "============================================================"
echo "ACT Training Job Submission"
echo "============================================================"
echo "Repo ID: $REPO_ID"
echo "Root: $ROOT"
echo "Environment: $ENV_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Training steps: $TRAINING_STEPS"
echo "============================================================"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Verify installation
echo "Verifying installation..."
python -c "import sir.training.train_act; print('✓ Installation verified')"
echo ""

# Set environment variables for training
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Build the training command
TRAIN_CMD="python -m sir.training.train_act --repo-id $REPO_ID --root $ROOT --batch-size $BATCH_SIZE --lr $LR --training-steps $TRAINING_STEPS --eval-freq $EVAL_FREQ --eval-episodes $EVAL_EPISODES --chunk-size $CHUNK_SIZE"

if [ "$SAVE_VIDEO" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --save-video"
fi

if [ "$USE_WANDB" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use-wandb --wandb-project $WANDB_PROJECT"
fi

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Run the training
eval "$TRAIN_CMD"

echo ""
echo "Training job completed!"
