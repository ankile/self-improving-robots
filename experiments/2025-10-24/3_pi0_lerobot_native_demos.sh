#!/bin/bash

# ============================================================================
# Fine-tune PI0 on square demo dataset using LeRobot's native training script
# ============================================================================
#
# Key Features:
# - Mixed Precision Training (AMP): Reduces memory usage by ~40% on 3090
# - No evaluation rollouts: Disabled to focus on training throughput
# - Uses LeRobot's Accelerator: Supports multi-GPU if needed later
# - Proper checkpoint management with resume support
#
# Memory Usage Notes:
# - Full precision (fp32) at batch_size=2: ~22GB (OOM on 3090)
# - Mixed precision (fp16/bf16) at batch_size=8: ~16GB (fits on 3090)
# - Can increase batch_size to 16-32 with AMP if needed
#
# To Resume Training:
# python -m lerobot.scripts.lerobot_train --config_path ./outputs/train/pi0_square_demos_amp/checkpoints/last/train_config.json --resume=true
#
# ============================================================================

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=ankile/square-v1 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.use_amp=true \
    --output_dir=./outputs/train/pi0_square_demos_amp \
    --job_name=pi0-demos-amp \
    --batch_size=8 \
    --steps=100000 \
    --eval_freq=0 \
    --log_freq=100 \
    --save_freq=5000 \
    --num_workers=4 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=square-dagger-comparison

# Notes on parameters:
# --policy.pretrained_path: Loads pretrained PI0 weights from HuggingFace
# --policy.use_amp: Enable automatic mixed precision (fp16/bf16) - REQUIRED for PI0!
# --job_name: Sets the wandb run name (not --wandb.name!)
# --batch_size: Start conservative at 8, can increase to 16-32 with AMP
# --eval_freq: Set to 0 to disable evaluation (or very high like 1000000)
# --save_freq: Save checkpoint every 5000 steps
# --num_workers: DataLoader workers for preprocessing (4 is usually good)
#
# Optional params to consider:
# --policy.device=cuda  # Explicit device selection
# --save_checkpoint=true  # Already true by default
# --use_policy_training_preset=true  # Use PI0's default optimizer/scheduler
# --wandb.notes="Description of this run"  # Add notes to the wandb run

