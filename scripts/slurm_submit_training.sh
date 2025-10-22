#!/bin/bash
#
# Convenient Wrapper for Submitting ACT Training Jobs to SLURM
#
# Usage:
#   bash scripts/slurm_submit_training.sh --repo-id REPO_ID --job-name JOB_NAME [OPTIONS]
#
# Examples:
#   # Basic training
#   bash scripts/slurm_submit_training.sh --repo-id my_dataset --job-name basic_training
#
#   # With custom parameters
#   bash scripts/slurm_submit_training.sh \
#     --repo-id my_dataset \
#     --job-name custom_training \
#     --batch-size 32 \
#     --training-steps 50000 \
#     --partition gpu \
#     --gpus 2 \
#     --time 120:00:00
#
#   # Dry run (show command without submitting)
#   bash scripts/slurm_submit_training.sh \
#     --repo-id my_dataset \
#     --job-name test \
#     --dry-run
#

set -e

# Defaults
REPO_ID=""
JOB_NAME="act-training"
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
PARTITION="gpu"
GPUS="1"
CPUS="8"
TIME="72:00:00"
MEM="64GB"
DRY_RUN="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
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
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --no-wandb)
            USE_WANDB="false"
            shift
            ;;
        --no-video)
            SAVE_VIDEO="false"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
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
    echo ""
    show_help
    exit 1
fi

# Create logs directory
mkdir -p logs

# Create a temporary SLURM script with the provided configuration
SLURM_SCRIPT=$(mktemp)
cat > "$SLURM_SCRIPT" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --gres=gpu:GPUS_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -e

echo "Starting ACT Training Job"
echo "=========================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (GPU info not available)"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ENV_NAME_PLACEHOLDER

# Run training
python -m sir.training.train_act \
  --repo-id REPO_ID_PLACEHOLDER \
  --root ROOT_PLACEHOLDER \
  --batch-size BATCH_SIZE_PLACEHOLDER \
  --lr LR_PLACEHOLDER \
  --training-steps TRAINING_STEPS_PLACEHOLDER \
  --eval-freq EVAL_FREQ_PLACEHOLDER \
  --eval-episodes EVAL_EPISODES_PLACEHOLDER \
  --chunk-size CHUNK_SIZE_PLACEHOLDER \
  SAVE_VIDEO_FLAG_PLACEHOLDER \
  WANDB_FLAG_PLACEHOLDER

echo ""
echo "Training job completed successfully!"
SLURM_EOF

# Replace placeholders in the temporary script
sed -i "s|JOB_NAME_PLACEHOLDER|$JOB_NAME|g" "$SLURM_SCRIPT"
sed -i "s|PARTITION_PLACEHOLDER|$PARTITION|g" "$SLURM_SCRIPT"
sed -i "s|CPUS_PLACEHOLDER|$CPUS|g" "$SLURM_SCRIPT"
sed -i "s|GPUS_PLACEHOLDER|$GPUS|g" "$SLURM_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|$TIME|g" "$SLURM_SCRIPT"
sed -i "s|MEM_PLACEHOLDER|$MEM|g" "$SLURM_SCRIPT"
sed -i "s|ENV_NAME_PLACEHOLDER|$ENV_NAME|g" "$SLURM_SCRIPT"
sed -i "s|REPO_ID_PLACEHOLDER|$REPO_ID|g" "$SLURM_SCRIPT"
sed -i "s|ROOT_PLACEHOLDER|$ROOT|g" "$SLURM_SCRIPT"
sed -i "s|BATCH_SIZE_PLACEHOLDER|$BATCH_SIZE|g" "$SLURM_SCRIPT"
sed -i "s|LR_PLACEHOLDER|$LR|g" "$SLURM_SCRIPT"
sed -i "s|TRAINING_STEPS_PLACEHOLDER|$TRAINING_STEPS|g" "$SLURM_SCRIPT"
sed -i "s|EVAL_FREQ_PLACEHOLDER|$EVAL_FREQ|g" "$SLURM_SCRIPT"
sed -i "s|EVAL_EPISODES_PLACEHOLDER|$EVAL_EPISODES|g" "$SLURM_SCRIPT"
sed -i "s|CHUNK_SIZE_PLACEHOLDER|$CHUNK_SIZE|g" "$SLURM_SCRIPT"

# Add optional flags
if [ "$SAVE_VIDEO" = "true" ]; then
    sed -i 's|SAVE_VIDEO_FLAG_PLACEHOLDER|--save-video|g' "$SLURM_SCRIPT"
else
    sed -i 's|SAVE_VIDEO_FLAG_PLACEHOLDER||g' "$SLURM_SCRIPT"
fi

if [ "$USE_WANDB" = "true" ]; then
    sed -i "s|WANDB_FLAG_PLACEHOLDER|--use-wandb --wandb-project $WANDB_PROJECT|g" "$SLURM_SCRIPT"
else
    sed -i 's|WANDB_FLAG_PLACEHOLDER||g' "$SLURM_SCRIPT"
fi

# Display job configuration
echo "============================================================"
echo "ACT Training Job Submission"
echo "============================================================"
echo ""
echo "Job Configuration:"
echo "  Job name: $JOB_NAME"
echo "  Repo ID: $REPO_ID"
echo "  Root: $ROOT"
echo ""
echo "Training Parameters:"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Training steps: $TRAINING_STEPS"
echo "  Eval frequency: $EVAL_FREQ"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Save video: $SAVE_VIDEO"
echo "  Use W&B: $USE_WANDB"
echo ""
echo "SLURM Configuration:"
echo "  Partition: $PARTITION"
echo "  GPUs: $GPUS"
echo "  CPUs: $CPUS"
echo "  Memory: $MEM"
echo "  Time: $TIME"
echo "  Env name: $ENV_NAME"
echo ""
echo "============================================================"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN - Job script created but not submitted:"
    echo ""
    cat "$SLURM_SCRIPT"
    echo ""
    rm "$SLURM_SCRIPT"
    exit 0
fi

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $NF}')
echo "✓ Job submitted with ID: $JOB_ID"
echo ""
echo "Monitor job status with:"
echo "  squeue -j $JOB_ID"
echo ""
echo "View job output with:"
echo "  tail -f logs/slurm-$JOB_ID.out"
echo ""

# Clean up temporary script
rm "$SLURM_SCRIPT"
