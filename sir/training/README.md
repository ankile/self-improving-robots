# Training Scripts

## train_act.py

Simple, straightforward training script for ACT (Action Chunking Transformer) policy using LeRobot.

### Features

- Loads LeRobotDataset from local storage
- Trains ACT policy with supervised behavioral cloning (BC) loss
- Evaluates in Robosuite environment at regular intervals
- Saves best model (by success rate) and periodic checkpoints
- Uses LeRobot's preprocessing/postprocessing for proper normalization

### Usage

Basic training:
```bash
python -m sir.training.train_act \
    --repo-id lift_minimal_state \
    --root ./data/test_filtered \
    --env Lift \
    --robot Panda
```

With custom parameters:
```bash
python -m sir.training.train_act \
    --repo-id lift_minimal_state \
    --root ./data/test_filtered \
    --env Lift \
    --robot Panda \
    --batch-size 32 \
    --lr 1e-5 \
    --training-steps 20000 \
    --eval-freq 2000 \
    --chunk-size 20
```

### Arguments

**Dataset:**
- `--repo-id`: Dataset name (required)
- `--root`: Root directory containing datasets (default: `./data`)

**Environment:**
- `--env`: Robosuite environment name (default: `Lift`)
- `--robot`: Robot name (default: `Panda`)

**Training:**
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-5)
- `--training-steps`: Number of training steps (default: 10000)
- `--eval-freq`: Evaluate every N steps (default: 10 for debugging, use 1000+ for real training)
- `--eval-episodes`: Episodes per evaluation (default: 10)
- `--max-steps`: Max steps per episode (default: 400)

**ACT Policy:**
- `--chunk-size`: Action chunk size (default: 20)
- `--n-obs-steps`: Number of observation steps (default: 1)
- `--n-action-steps`: Actions to execute per chunk (default: 20)

**System:**
- `--device`: Device to use - cuda/mps/cpu (auto-detected)
- `--num-workers`: Dataloader workers (default: 4)
- `--checkpoint-dir`: Checkpoint directory (default: `./checkpoints`)

### Output

Checkpoints are saved to `./checkpoints/{repo_id}/`:
- `best_model/`: Best model by success rate
- `checkpoint_{step}/`: Periodic checkpoints (every 5000 steps)
- `final_model/`: Final model after training

Each checkpoint includes:
- Policy weights (`policy.safetensors`)
- Preprocessor (`preprocessor/`)
- Postprocessor (`postprocessor/`)
- Configuration (`config.json`)

### Example Training Run

```bash
# Train on collected Lift demonstrations
python -m sir.training.train_act \
    --repo-id lift_minimal_state \
    --root ./data/test_filtered \
    --env Lift \
    --robot Panda \
    --training-steps 10000 \
    --eval-freq 1000 \
    --eval-episodes 10
```

Output:
```
============================================================
Training ACT Policy with LeRobot
============================================================
Dataset: lift_minimal_state
Root: ./data/test_filtered
Environment: Lift (Panda)
Device: mps
Batch size: 16
Learning rate: 1e-05
Training steps: 10000
Action chunk size: 20
============================================================

Loading dataset metadata...
✓ Dataset metadata loaded
  Total episodes: 6
  Total frames: 845
  FPS: 20

Input features:
  observation.state: shape=(120,)
Output features:
  action: shape=(7,)

Creating ACT policy...
✓ ACT policy created
  Parameters: 6,234,567

Loading dataset...
✓ Dataset loaded: 845 frames

Creating evaluation environment...
✓ Evaluation environment created

Checkpoints will be saved to: ./checkpoints/lift_minimal_state

Starting training...
============================================================
step: 0/10000 loss: 12.3456 l1: 12.3456 kl: 0.0000
step: 100/10000 loss: 8.9012 l1: 8.9012 kl: 0.0012
...

Evaluating policy...
Evaluation Results (step 1000):
  Success Rate: 20.0%
  Avg Reward: 45.234
  Avg Length: 156.3

✓ Saved best model (success rate: 20.0%)
...
```

### Notes

- ACT uses action chunking: predicts multiple future actions at once
- Default chunk size is 20 (good balance between temporal consistency and responsiveness)
- Training uses VAE loss (L1 + KL divergence) by default
- Evaluation runs full episodes in the actual Robosuite environment
- Best model is selected based on success rate during evaluation
- Images are automatically detected and processed through ResNet18 vision backbone
- State observations are concatenated with image features for the policy
