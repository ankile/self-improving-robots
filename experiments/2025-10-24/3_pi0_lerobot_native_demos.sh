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
# - AGGRESSIVE MEMORY OPTIMIZATION: Multiple strategies to fit on 24GB 3090
# - Mixed Precision Training (AMP): Use bfloat16 dtype + amp
# - Gradient Checkpointing: Trade compute for memory (critical!)
# - Smaller Model: Use gemma_300m instead of gemma_2b
# - Reduced Image Resolution: 128x128 instead of 224x224
# - Reduced Chunk Size: 25 instead of 50
# - Zero DataLoader Workers: No multiprocessing memory overhead
#
# Memory Optimization Breakdown:
# 1. dtype=bfloat16 + use_amp: ~30-40% memory reduction
# 2. gradient_checkpointing=true: ~40-50% memory reduction (huge!)
# 3. gemma_300m vs gemma_2b: ~75% model parameter reduction
# 4. image_resolution=(128,128): ~66% image memory reduction
# 5. chunk_size=25: 50% action sequence memory reduction
# 6. num_workers=0: Eliminates multiprocessing memory overhead
# 7. batch_size=1: Minimal per-batch memory
#
# Expected Memory Usage: ~12-16GB (should fit on 3090!)
#
# To Resume Training:
# python -m lerobot.scripts.lerobot_train --config_path ./outputs/train/pi0_square_ultra_low_mem/checkpoints/last/train_config.json --resume=true
#
# ============================================================================

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=ankile/square-v1 \
    --policy.type=pi0 \
    --policy.use_amp=true \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.paligemma_variant=gemma_300m \
    --policy.action_expert_variant=gemma_300m \
    --policy.image_resolution="(128, 128)" \
    --policy.chunk_size=25 \
    --policy.n_action_steps=25 \
    --policy.push_to_hub=false \
    --output_dir=./outputs/train/pi0_square_ultra_low_mem \
    --job_name=pi0-ultra-low-mem \
    --batch_size=1 \
    --steps=100000 \
    --eval_freq=0 \
    --log_freq=100 \
    --save_freq=5000 \
    --num_workers=0 \
    --seed=1000 \
    --wandb.enable=true \
    --wandb.project=square-dagger-comparison

# Notes on Memory Optimization Parameters:
#
# CRITICAL MEMORY SAVERS:
# --policy.gradient_checkpointing=true: THE MOST IMPORTANT! Trades compute for memory
#                                        Saves activations by recomputing during backward pass
#                                        Can reduce memory by 40-50% but increases training time ~20-30%
#
# --policy.dtype=bfloat16: Sets model weights to bfloat16 (works with use_amp)
#                           bfloat16 has better numerical stability than fp16
#                           Required when using use_amp for mixed precision
#
# --policy.paligemma_variant=gemma_300m: Use smaller vision encoder (300M vs 2B params)
# --policy.action_expert_variant=gemma_300m: Use smaller action decoder (300M vs 2B params)
#                                             Combined: ~75% reduction in model size
#
# --policy.image_resolution="(128, 128)": Reduce from default 224x224
#                                          ~66% reduction in image memory
#                                          May impact visual precision, test performance
#
# --policy.chunk_size=25: Reduce from default 50
# --policy.n_action_steps=25: Must match chunk_size
#                              50% reduction in action sequence memory
#                              Shorter prediction horizon may impact long-term planning
#
# --num_workers=0: Disable DataLoader multiprocessing
#                   Eliminates memory overhead from worker processes
#                   Slower data loading but worth it for memory-constrained systems
#
# --batch_size=1: Minimum batch size
#                  Linear memory scaling - batch_size=2 would double batch memory
#
# OTHER PARAMETERS:
# --policy.type=pi0: Train from scratch (not loading pretrained weights)
# --policy.use_amp=true: Enable automatic mixed precision training
# --policy.push_to_hub=false: Disable auto-pushing to HuggingFace Hub
# --job_name: Sets the wandb run name
# --eval_freq=0: Disable evaluation rollouts to save memory
# --save_freq=5000: Save checkpoint every 5000 steps
#
# TRADE-OFFS:
# - gradient_checkpointing: More compute, slower training, but much less memory
# - Smaller models (300m): Less capacity, may impact performance on complex tasks
# - Lower resolution (128): Less visual detail, may hurt vision-dependent tasks
# - Smaller chunk_size (25): Shorter planning horizon, may impact long-horizon tasks
# - num_workers=0: Slower data loading, may bottleneck training
#
# IF STILL GETTING OOM:
# 1. Reduce image_resolution to (96, 96) or (64, 64)
# 2. Reduce chunk_size to 10 or 15
# 3. Check for memory leaks: nvidia-smi to monitor GPU memory over time
# 4. Verify no other processes using GPU: nvidia-smi
# 5. Try training on a machine with more GPU memory (A100 40GB or 80GB)

