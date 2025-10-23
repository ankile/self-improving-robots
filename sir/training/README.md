# Training Scripts

## train_policy.py

General training script for robot learning policies (ACT, Diffusion, PI0, etc.) using LeRobot.

### Features

- **Multi-policy support**: Train ACT, Diffusion, PI0, PI05, VQBeT, TDMPC, SAC, or SmolVLA policies
- **Multi-dataset support**: Automatically combines multiple datasets (e.g., BC + DAgger)
- **DAgger filtering**: Automatically filters DAgger datasets to successful human corrections
- Loads LeRobotDataset from local storage or HuggingFace Hub
- Trains with supervised behavioral cloning (BC) loss
- Evaluates in Robosuite environment at regular intervals
- Saves best model (by success rate) and periodic checkpoints
- Uses LeRobot's preprocessing/postprocessing for proper normalization
- Optional Weights & Biases logging and video recording

### Usage

Train ACT policy (default):
```bash
python -m sir.training.train_policy \
    --repo-ids lift_minimal_state \
    --root ./data \
    --policy act
```

Train Diffusion policy:
```bash
python -m sir.training.train_policy \
    --repo-ids lift_demos \
    --root ./data \
    --policy diffusion
```

Train with multiple datasets (BC + DAgger):
```bash
python -m sir.training.train_policy \
    --repo-ids "lift_demos,lift_dagger" \
    --root ./data \
    --policy act
```

With custom parameters:
```bash
python -m sir.training.train_policy \
    --repo-ids lift_demos \
    --root ./data \
    --policy diffusion \
    --batch-size 32 \
    --lr 1e-4 \
    --training-steps 20000 \
    --eval-freq 2000 \
    --action-chunk-size 16
```

### Arguments

**Policy:**
- `--policy`: Policy type - act/diffusion/pi0/pi05/vqbet/tdmpc/sac/smolvla (default: act)

**Dataset:**
- `--repo-ids`: Comma-separated dataset names (required)
- `--root`: Root directory containing datasets (default: HuggingFace Hub cache)

**Environment:**
- `--env`: Robosuite environment name (default: `Lift`)
- `--robot`: Robot name (default: `Panda`)

**Training:**
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: policy-specific default)
- `--training-steps`: Number of training steps (default: 10000)
- `--eval-freq`: Evaluate every N steps (default: 1000)
- `--eval-episodes`: Episodes per evaluation (default: 10)
- `--max-steps`: Max steps per episode (default: 400)

**Policy-Specific:**
- `--action-chunk-size`: Action chunk/horizon size (default: policy-specific)
- `--n-obs-steps`: Number of observation steps (default: policy-specific)
- `--n-action-steps`: Actions to execute per chunk (default: policy-specific)

**System:**
- `--device`: Device to use - cuda/mps/cpu (auto-detected)
- `--num-workers`: Dataloader workers (default: 4)
- `--checkpoint-dir`: Checkpoint directory (default: `./checkpoints`)

**Logging:**
- `--use-wandb`: Enable Weights & Biases logging
- `--wandb-project`: W&B project name (default: `act-training`)
- `--save-video`: Save evaluation rollout videos

### Output

Checkpoints are saved to `./checkpoints/{policy}_{dataset_name}/`:
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
# Train ACT policy on Lift demonstrations
python -m sir.training.train_policy \
    --repo-ids lift_demos \
    --root ./data \
    --policy act \
    --training-steps 10000 \
    --eval-freq 1000
```

Output:
```
============================================================
Training ACT Policy with LeRobot
============================================================
Policy: act
Datasets: lift_demos
Root: ./data
Environment: Lift (Panda)
Device: mps
Batch size: 16
============================================================

Filtering datasets...
Processing dataset: lift_demos
  Dataset 'lift_demos' is not a DAgger dataset (no source/success columns)
  Using all 6 episodes

Loading dataset metadata...
✓ Dataset metadata loaded
  FPS: 20

Input features:
  observation.state: shape=(120,)
Output features:
  action: shape=(7,)

Creating ACT policy...
✓ ACT policy created
  Parameters: 6,234,567
  Chunk size: 20
  Observation steps: 1
  Action steps: 20
  Learning rate: 1e-05

Loading dataset(s)...
✓ Datasets loaded:
  lift_demos: 6 episodes (of 6 total)
  Total frames: 845

Creating evaluation environment...
✓ Evaluation environment created

Checkpoints will be saved to: ./checkpoints/act_lift_demos

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

### Supported Policies

- **ACT** (Action Chunking Transformer): Uses action chunking and VAE loss (L1 + KL divergence)
- **Diffusion**: Diffusion-based action prediction with temporal consistency
- **PI0/PI05**: Physical Intelligence's VLA-based policies
- **VQBeT**: Vector-Quantized Behavior Transformer
- **TDMPC**: Temporal Difference Model Predictive Control
- **SAC**: Soft Actor-Critic (for RL fine-tuning)
- **SmolVLA**: Efficient vision-language-action model

### Notes

- Each policy uses its own default hyperparameters (learning rate, chunk size, etc.)
- You can override defaults with command-line arguments
- DAgger datasets are automatically filtered to successful human corrections
- Multiple datasets are combined on-the-fly during training
- Evaluation runs full episodes in the actual Robosuite environment
- Best model is selected based on success rate during evaluation
- Camera observations are automatically detected and processed
- State observations are properly normalized using dataset statistics
