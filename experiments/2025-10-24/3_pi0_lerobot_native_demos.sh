#!/bin/bash

# ============================================================================
# Train PI0 from scratch on square demo dataset using LeRobot's native training script
# ============================================================================
#
# Note: Cannot use pretrained lerobot/pi0_base because camera names don't match:
#   - PI0 base expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
#   - Our dataset has: agentview, robot0_eye_in_hand
# Therefore, training from scratch on our dataset.
#
# Key Features:
# - Mixed Precision Training (AMP): Reduces memory usage by ~40% on 3090
# - No evaluation rollouts: Disabled to focus on training throughput
# - Uses LeRobot's Accelerator: Supports multi-GPU if needed later
# - Proper checkpoint management with resume support
#
# Memory Usage Notes:
# - Full precision (fp32) at batch_size=1: >24GB (OOM on 3090)
# - Mixed precision (fp16/bf16) at batch_size=8: ~16GB (fits on 3090)
# - Can increase batch_size to 16-32 with AMP if needed
#
# To Resume Training:
# python -m lerobot.scripts.lerobot_train --config_path ./outputs/train/pi0_square_scratch_amp/checkpoints/last/train_config.json --resume=true
#
# ============================================================================

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=ankile/square-v1 \
    --policy.type=pi0 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --output_dir=./outputs/train/pi0_square_scratch_amp \
    --job_name=pi0-scratch-amp \
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
# --policy.type=pi0: Train PI0 from scratch (not loading pretrained weights)
#                    Use --policy.path=lerobot/pi0_base to load pretrained, but that requires
#                    matching camera names (base_0_rgb, left/right_wrist_0_rgb)
# --policy.use_amp: Enable automatic mixed precision (fp16/bf16) - REQUIRED for PI0!
# --policy.push_to_hub: Set to false to disable auto-pushing to HuggingFace Hub
#                       Default is true, which requires --policy.repo_id
# --job_name: Sets the wandb run name (not --wandb.name!)
# --batch_size: Start conservative at 8, can increase to 16-32 with AMP
# --eval_freq: Set to 0 to disable evaluation (or very high like 1000000)
# --save_freq: Save checkpoint every 5000 steps
# --num_workers: DataLoader workers for preprocessing (4 is usually good)
#
# Why training from scratch?
# The pretrained lerobot/pi0_base model expects different camera names than our dataset.
# To use pretrained weights, you would need to:
#   1. Remap camera names in your dataset to match PI0's expected names, OR
#   2. Use a dataset that already has matching camera names
#
# Optional params to consider:
# --policy.device=cuda  # Explicit device selection
# --save_checkpoint=true  # Already true by default
# --use_policy_training_preset=true  # Use PI0's default optimizer/scheduler
# --wandb.notes="Description of this run"  # Add notes to the wandb run

