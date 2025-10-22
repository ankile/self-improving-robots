# W&B Model Artifacts Guide

This guide explains how to use the W&B artifact system for model versioning and the BC → DAgger → RL pipeline.

## Overview

The training pipeline now automatically saves model checkpoints to Weights & Biases, creating a complete artifact pipeline:

```
BC Training (train_act.py)
    ↓ saves best_model to W&B
    ↓
W&B Artifact: act-dataset-best-step-N
    ↓ download for DAgger
    ↓
DAgger Training (train_dagger.py - future)
    ↓ fine-tunes from BC checkpoint
    ↓
W&B Artifact: dagger-dataset-best-step-N
    ↓ download for RL
    ↓
RL Training (train_rl.py - future)
    ↓ continues learning from DAgger
    ↓
W&B Artifact: rl-dataset-final
```

## What Gets Saved

When a checkpoint is saved during training, **all 6 files** are uploaded to W&B:

```
artifact_directory/
├── config.json                                    # Policy architecture
├── model.safetensors                             # Neural network weights (~197 MB)
├── policy_preprocessor.json                      # Input normalization config
├── policy_preprocessor_step_3_normalizer_processor.safetensors  # Input stats
├── policy_postprocessor.json                     # Output denormalization config
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors # Output stats
```

This ensures that policies can be loaded with **identical normalization** across machines and training phases.

## Checkpoint Timing

The training script uploads checkpoints at:

1. **Best Model** - When evaluation shows improved success rate
   - Artifact name: `act-{dataset}-best-step-{N}`
   - Metadata: success_rate, avg_reward, step, is_best=true

2. **Periodic Checkpoints** - Every 5000 training steps
   - Artifact name: `act-{dataset}-checkpoint-step-{N}`
   - Useful for analyzing training progress

3. **Final Model** - After training completes
   - Artifact name: `act-{dataset}-final`
   - Final evaluation results included

## Loading Policies

### Option 1: Load from Local Checkpoint

```python
from sir.training.load_pretrained import load_policy_from_checkpoint

policy, preprocessor, postprocessor = load_policy_from_checkpoint(
    checkpoint_path="checkpoints/lift_v1/best_model",
    device="cuda"
)

# Use for inference
observations = {...}
actions = policy(observations)
```

### Option 2: Load from W&B (Recommended for Sharing)

```python
from sir.training.load_pretrained import load_policy_from_wandb
import wandb

# Must initialize W&B first
wandb.init(project="my-project")

# Load from artifact
policy, preprocessor, postprocessor = load_policy_from_wandb(
    artifact_identifier="act-lift-best:latest",
    device="cuda"
)
```

### Option 3: Load BC Policy for DAgger (Recommended)

```python
from sir.training.load_pretrained import load_bc_policy_for_dagger

# From local checkpoint
policy, prep, post = load_bc_policy_for_dagger(
    checkpoint_source="checkpoints/lift/best_model"
)

# Or from W&B
policy, prep, post = load_bc_policy_for_dagger(
    checkpoint_source="act-lift-best-step-10000:latest",
    from_wandb=True,
    device="cuda"
)

# Use for DAgger training initialization...
```

## Using BC Policies for DAgger

Here's how to use a trained BC policy as initialization for DAgger training:

```python
import wandb
from sir.training.load_pretrained import load_bc_policy_for_dagger
from sir.training.dagger import DAggerTrainer

# Initialize W&B
wandb.init(
    project="my-project",
    job_type="dagger",
    tags=["dagger", "from-bc"]
)

# Load BC policy
print("Loading BC policy...")
bc_policy, preprocessor, postprocessor = load_bc_policy_for_dagger(
    checkpoint_source="act-lift-best-step-10000:latest",
    from_wandb=True,
    device="cuda"
)

# Initialize DAgger trainer with BC policy
trainer = DAggerTrainer(
    initial_policy=bc_policy,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    num_human_interventions=100,
    intervention_frequency=0.1,  # Request intervention 10% of the time
)

# Train with interventions
trainer.train(
    dataset_repo_id="my-dataset",
    num_episodes=500,
    max_steps_per_episode=500,
)

# Best DAgger policy is automatically saved to:
# - Local: checkpoints/lift_v1/best_model
# - W&B: dagger-lift-best-step-N
```

## Available Utilities

### List All BC Policies

```python
from sir.training.load_pretrained import list_available_bc_policies

policies = list_available_bc_policies()
for p in policies:
    print(f"{p['name']:40} | {p['metadata']['success_rate']:>6} | {p['created_at']}")
```

### Get Policy Metadata

```python
import wandb

api = wandb.Api()
artifact = api.artifact("act-lift-best-step-10000:latest")

print("Artifact Metadata:")
print(f"  Success Rate: {artifact.metadata['success_rate']}")
print(f"  Avg Reward: {artifact.metadata['avg_reward']}")
print(f"  Step: {artifact.metadata['step']}")
print(f"  Dataset: {artifact.metadata['repo_id']}")
```

## File Structure

### Uploaded Files in W&B

All artifacts follow the same structure after download:

```
~/.cache/sir_wandb_artifacts/act-lift-best-step-10000/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

### Local Checkpoints

Training saves checkpoints locally at:

```
checkpoints/
└── {dataset}/
    ├── best_model/           (best checkpoint during training)
    ├── checkpoint_5000/      (periodic)
    ├── checkpoint_10000/     (periodic)
    └── final_model/          (final after training)
```

Each checkpoint directory contains the same 6 files.

## Training with W&B

### Submit Training with Artifact Saving

```bash
python -m sir.training.train_act \
  --repo-id my-dataset \
  --use-wandb \
  --wandb-project my-project \
  --batch-size 32 \
  --training-steps 50000
```

During training, you'll see:

```
step: 1500/50000 loss: 2.1234 l1: 0.4567
  Saved best model (success rate: 85.0%)
  ✓ Artifact 'act-my-dataset-best-step-1500' uploaded successfully

step: 5000/50000 loss: 1.8902 l1: 0.3421
  ✓ Saved checkpoint at step 5000
  ✓ Artifact 'act-my-dataset-checkpoint-step-5000' uploaded successfully
```

### Monitor Artifacts in W&B

1. Go to your W&B project: `wandb.ai/your-entity/your-project`
2. Click on a run to view run details
3. Scroll to "Model" or "Artifacts" section
4. See all uploaded checkpoints with metadata

## Best Practices

### 1. Use Descriptive Dataset Names

```python
# Good
--repo-id lift-v1-with-10k-demos
--repo-id stack-cube-bimanual-v2

# Avoid
--repo-id dataset1
--repo-id data
```

The artifact name includes the dataset name, so it's easier to track.

### 2. Track Pipeline Stages

Use W&B job types to track the pipeline:

```bash
# BC training
python -m sir.training.train_act \
  --repo-id lift-demos \
  --use-wandb \
  --wandb-project lift-pipeline \
  --wandb-job-type bc-training

# DAgger (future)
python -m sir.training.train_dagger \
  --policy-artifact "act-lift-best:latest" \
  --use-wandb \
  --wandb-project lift-pipeline \
  --wandb-job-type dagger-training

# RL (future)
python -m sir.training.train_rl \
  --policy-artifact "dagger-lift-best:latest" \
  --use-wandb \
  --wandb-project lift-pipeline \
  --wandb-job-type rl-training
```

### 3. Document Hyperparameters

The full training config is automatically logged to W&B, but add notes:

```bash
python -m sir.training.train_act \
  --repo-id lift-demos \
  --batch-size 32 \
  --lr 5e-05 \
  --use-wandb \
  --wandb-notes "Testing increased LR (5e-05 vs 1e-05)"
```

### 4. Archive Old Artifacts

Keep only recent versions to save quota:

```python
import wandb

api = wandb.Api()
artifacts = list(api.artifacts(type_name="policy"))

# Keep only last 5 versions
if len(artifacts) > 5:
    for a in artifacts[:-5]:
        a.delete()
```

## Troubleshooting

### Artifact Upload Fails

**Problem**: `Failed to upload artifact`

**Solutions**:
1. Check W&B is logged in: `wandb login`
2. Check internet connection during training
3. Check storage quota hasn't been exceeded

### Cannot Load Artifact

**Problem**: `Artifact not found`

**Solutions**:
1. Check artifact name: `wandb artifact list {project}`
2. Verify version exists (use `:latest` if unsure)
3. Ensure you have access to the project

### Model Loads but Inference Fails

**Problem**: `Shape mismatch` or `RuntimeError during inference`

**Solutions**:
1. Ensure correct device: `policy.to(device)`
2. Check preprocessor is applied to observations
3. Verify dataset compatibility (same features)

## API Reference

### wandb_artifacts.py

```python
upload_checkpoint_to_wandb(
    checkpoint_path: Path,
    artifact_name: str,
    artifact_type: str = "policy",
    description: Optional[str] = None,
    metadata: Optional[dict] = None
) -> wandb.Artifact

download_checkpoint_from_wandb(
    artifact_identifier: str,
    download_dir: Optional[Path] = None
) -> Path

list_checkpoints_from_wandb(
    artifact_type: str = "policy",
    project: Optional[str] = None
) -> list[dict]

create_artifact_metadata(
    repo_id: str,
    success_rate: float,
    avg_reward: float,
    step: int,
    is_best: bool = False,
    dataset_size: int = 0,
    checkpoint_index: Optional[int] = None
) -> dict
```

### load_pretrained.py

```python
load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict: bool = False
) -> Tuple[ACTPolicy, dict, dict]

load_policy_from_wandb(
    artifact_identifier: str,
    device: str = "cpu",
    strict: bool = False
) -> Tuple[ACTPolicy, dict, dict]

load_bc_policy_for_dagger(
    checkpoint_source: str | Path,
    device: str = "cuda",
    from_wandb: bool = False
) -> Tuple[ACTPolicy, dict, dict]

list_available_bc_policies() -> list[dict]
```

## Examples

### Complete BC to DAgger Pipeline

```python
import wandb
from pathlib import Path
from sir.training.load_pretrained import load_bc_policy_for_dagger

# 1. Train BC policy
print("=" * 60)
print("Step 1: Training BC policy")
print("=" * 60)
# Run train_act.py with --use-wandb
# Artifacts saved to W&B automatically

# 2. Load BC policy for DAgger
print("\n" + "=" * 60)
print("Step 2: Loading BC policy for DAgger")
print("=" * 60)

wandb.init(project="lift-pipeline", job_type="dagger")

policy, prep, post = load_bc_policy_for_dagger(
    checkpoint_source="act-lift-demos-best:latest",
    from_wandb=True,
    device="cuda"
)
print(f"✓ Loaded BC policy: {policy.config}")
print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

# 3. DAgger training would happen here
print("\n" + "=" * 60)
print("Step 3: DAgger training (not yet implemented)")
print("=" * 60)
print("Next: Implement sir.training.train_dagger")
print("It will use the loaded policy and preprocessor/postprocessor")
```

## Future Work

The artifact system is designed to support:

1. **DAgger Training** - Load BC checkpoints, fine-tune with corrections
2. **RL Fine-tuning** - Load DAgger checkpoints, continue with RL
3. **Model Ensembles** - Combine multiple checkpoints
4. **Ablation Studies** - Compare different training stages
5. **Cross-Dataset Transfer** - Load policies trained on similar tasks

## See Also

- `sir/training/train_act.py` - BC training script (auto-saves artifacts)
- `sir/training/wandb_artifacts.py` - W&B utilities
- `sir/training/load_pretrained.py` - Policy loading utilities
- WANDB Docs: https://docs.wandb.ai/guides/artifacts
- LeRobot Docs: https://github.com/lerobotics/lerobot
