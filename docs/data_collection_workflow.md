# Data Collection Workflow Design

This document outlines the design considerations and recommended workflow for collecting robosuite teleoperation data and converting it to LeRobot dataset format for iterative learning experiments.

## Context

Our research focuses on continual robot learning through BC → DAgger → RL progression. Key questions:
- When does BC saturate vs. reach 100% success?
- What is the scaling crossover point where DAgger/RL becomes beneficial?
- How to track data provenance across multiple collection rounds?

## Design Decision: Separate Rounds + Explicit Processing

**Recommendation: Keep separate datasets per collection round, with explicit merge/filter steps before training.**

### Rationale

1. **Provenance Tracking** - Essential for answering "how much data from which round?" in ablation studies
2. **Flexible Training Recipes** - Mix and match datasets for different experiments without re-collecting
3. **Data Quality Control** - Review and filter each round before merging
4. **Reproducibility** - Clear lineage from raw episodes → filtered → merged → trained

## Directory Structure

```
data/
├── raw_episodes/                        # Intermediate storage (can delete after conversion)
│   └── lift/
│       ├── round_0_bc_demos/
│       │   ├── episode_000000/
│       │   │   ├── episode_data.npz
│       │   │   ├── agentview/
│       │   │   └── robot0_eye_in_hand/
│       │   └── ...
│       └── round_1_policy_rollouts/
│           └── ...
├── lerobot_datasets/                    # LeRobot format (ready for training)
│   ├── lift_round_0_bc_demos/          # Initial 50 human demos
│   ├── lift_round_1_policy_rollouts/   # 100 policy rollouts (some failures)
│   ├── lift_round_2_corrections/       # 25 human corrections on failures
│   ├── lift_round_3_rl_finetune/       # 200 RL exploration episodes
│   └── lift_aggregated_v1/             # Merged dataset for specific experiment
```

## Four-Phase Workflow

### Phase 1: Collection (Per Round)

```bash
# Round 0: Initial demonstrations
mjpython -m sir.teleoperation.robosuite_teleop \
    --env Lift --robot Panda \
    --save-data \
    --round-id "round_0_bc_demos" \
    --target-episodes 50 \
    --cameras "agentview,robot0_eye_in_hand" \
    --output-dir ./data/raw_episodes/lift

# Later rounds: Policy rollouts, corrections, RL exploration
mjpython -m sir.teleoperation.robosuite_teleop \
    --env Lift --robot Panda \
    --save-data \
    --round-id "round_2_corrections" \
    --parent-policy "lift_bc_v1" \
    --output-dir ./data/raw_episodes/lift
```

**Additions needed to `robosuite_teleop.py`:**
- `--save-data` flag to enable recording
- `--round-id` for provenance tracking
- `--parent-policy` to track which policy generated rollouts
- Episode metadata: task name, success flag, timestamps
- Save to structured format (NPZ for non-images, PNG/NPZ for cameras)

### Phase 2: Conversion (Per Round)

```bash
# Convert raw episodes to LeRobot format
python -m sir.data.convert_robosuite_to_lerobot \
    --input-dir data/raw_episodes/lift/round_0_bc_demos \
    --output-dir data/lerobot_datasets/lift_round_0_bc_demos \
    --metadata round=0,source=human_demo,task=lift
```

**Script responsibilities:**
- Load raw episode data (NPZ, images)
- Map to LeRobot format:
  - `observation.state` ← Robot joint positions/velocities
  - `observation.images.{camera_name}` ← Camera frames as [C, H, W] tensors
  - `action` ← 7-DOF actions [dx, dy, dz, dpitch, droll, dyaw, gripper]
  - `timestamp` ← Frame timing
  - `task` ← Task name
- Encode videos to MP4 (LeRobot v3.0 format)
- Generate dataset metadata automatically:
  - Success rate (from reward/done flags)
  - Average episode length
  - Collection date and parameters

### Phase 3: Review & Filter (Per Round)

```bash
# Visualize episodes to check quality
python -m sir.data.review_episodes \
    data/lerobot_datasets/lift_round_0_bc_demos \
    --show-failures

# Filter low-quality episodes
python -m sir.data.filter_dataset \
    lift_round_0_bc_demos \
    --min-success-reward 0.8 \
    --max-episode-length 200 \
    --output lift_round_0_bc_demos_filtered
```

**Utility functions:**
- Episode playback with camera views
- Statistics per episode (reward, length, success)
- Filtering by various criteria
- Outlier detection

### Phase 4: Training (Mix & Match)

**Option A: Multi-dataset training (recommended)**
```python
# LeRobot v3.0 supports multiple datasets natively
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

datasets = [
    LeRobotDataset("lift_round_0_bc_demos"),
    LeRobotDataset("lift_round_2_corrections"),
]
# Training code handles concatenation automatically
```

**Option B: Explicit merge**
```bash
# Create merged version for specific experiment
python -m sir.data.merge_datasets \
    lift_round_0_bc_demos \
    lift_round_2_corrections \
    --output lift_bc_plus_dagger_v1 \
    --metadata "experiment=scaling_crossover_bc_to_dagger"
```

## Dataset Metadata Schema

Each dataset should store rich metadata for reproducibility:

```json
{
  "dataset_id": "lift_round_0_bc_demos",
  "task": "Lift",
  "robot": "Panda",
  "environment": "robosuite-v1.4",
  "round_id": 0,
  "collection_type": "human_demo",  // or "policy_rollout", "correction", "rl_explore"
  "parent_policy": null,  // or "lift_bc_v1_checkpoint_1000" for rollouts
  "collection_date": "2025-10-17",
  "num_episodes": 50,
  "success_rate": 0.92,
  "avg_episode_length": 127,
  "total_frames": 6350,
  "collection_params": {
    "controller": "OSC_POSE",
    "control_freq": 20,
    "cameras": ["agentview", "robot0_eye_in_hand"],
    "camera_resolution": [256, 256],
    "action_dim": 7,
    "state_dim": 32
  },
  "statistics": {
    "action": {"min": [...], "max": [...], "mean": [...], "std": [...]},
    "observation.state": {"min": [...], "max": [...], "mean": [...], "std": [...]}
  }
}
```

## Research-Specific Use Cases

### BC Saturation Analysis
```python
# Easy to create training curves with different data amounts
for n in [10, 25, 50, 100, 200]:
    train_policy(
        dataset="lift_round_0_bc_demos",
        num_episodes=n,
        experiment_name=f"bc_saturation_n{n}"
    )
```

### Scaling Crossover (BC → DAgger)
```python
# Compare: BC-only vs BC+DAgger with same total trajectory count
configs = [
    {"datasets": ["round_0[:50]"], "name": "bc_50"},
    {"datasets": ["round_0[:25]", "round_1[:25]"], "name": "dagger_25+25"},
    {"datasets": ["round_0[:25]", "round_2[:25]"], "name": "bc_corrections_25+25"},
]
```

### RL Fine-tuning from BC Initialization
```python
# Track lineage: demos → BC policy → RL rollouts
policy = load_pretrained("lift_bc_v1")
collect_rl_data(
    policy=policy,
    output="lift_round_3_rl_finetune",
    metadata={
        "parent_policy": "lift_bc_v1",
        "rl_algo": "sac",
        "exploration_noise": 0.2
    }
)
```

## LeRobot Integration Details

### LeRobot v3.0 Key Features
- **Multiple episodes per Parquet file** - Efficient storage, no filesystem limits
- **Streaming support** - Train on large datasets without downloading
- **MP4 video encoding** - More efficient than PNG sequences
- **Rich metadata** - Episode-level queries and filtering

### LeRobot Dataset Creation API
```python
from lerobot.common.datasets.populate_dataset import (
    init_dataset,
    add_frame,
    save_episode,
    encode_videos,
    save_lerobot_dataset_on_disk,
    push_lerobot_dataset_to_hub
)

# Initialize
dataset = init_dataset(...)

# Per episode
for episode in episodes:
    for frame in episode:
        dataset = add_frame(dataset, frame)
    dataset = save_episode(dataset)

# Finalize
dataset = encode_videos(dataset)
save_lerobot_dataset_on_disk(dataset, output_dir)
push_lerobot_dataset_to_hub(dataset, repo_id)  # Optional
```

### LeRobot Dataset Merging
LeRobot provides built-in merge functionality:
- Script: `lerobot/common/datasets/merge.py`
- Handles different dimensions with padding
- Preserves data integrity and episode indices
- **Limitation**: Only datasets with same FPS can be merged

## Implementation Priorities

1. **Immediate** - Update `robosuite_teleop.py`:
   - Add `--save-data`, `--round-id`, `--parent-policy` flags
   - Implement episode saving to intermediate format (NPZ + images)
   - Add episode-level metadata collection

2. **Next** - Create `sir/data/convert_robosuite_to_lerobot.py`:
   - Load raw episodes
   - Map to LeRobot format
   - Generate statistics and metadata
   - Encode videos to MP4

3. **Then** - Dataset utilities:
   - `merge_datasets.py` - Wrapper around LeRobot's merge with metadata handling
   - `filter_dataset.py` - Quality control and episode selection
   - `review_episodes.py` - Visualization and statistics

4. **Later** - Advanced features:
   - Automated quality scoring
   - Active learning episode selection
   - Dataset versioning with DVC
   - Train/test split generation

## Storage Estimates

- **Raw episodes** (before encoding): ~50-100 MB per episode with cameras
- **LeRobot format** (after MP4 encoding): ~1-5 GB per 100 episodes
- **Recommendation**: Delete raw episodes after successful conversion to save space

## Robosuite-Specific Considerations

### State Observation Extraction
Robosuite observations vary by environment:
- Robot joint positions: `obs["robot0_joint_pos"]`
- Robot joint velocities: `obs["robot0_joint_vel"]`
- Gripper state: `obs["robot0_gripper_qpos"]`
- Object positions: `obs["{object_name}_pos"]` (if available)
- End-effector pose: `obs["robot0_eef_pos"]`, `obs["robot0_eef_quat"]`

Need flexible extraction based on available keys.

### Camera Observations
- Format: `obs["{camera_name}_image"]` as numpy array (H, W, 3)
- Common cameras: `agentview`, `robot0_eye_in_hand`, `birdview`
- Convert to torch tensors with [C, H, W] layout for LeRobot

### Bimanual Robots
- Double action/state dimensions
- Actions: `[robot0_action, robot1_action]`
- Need special handling in conversion script

### Episode Success Detection
- Robosuite provides `done` flag but not always success
- Use reward threshold or task-specific success criteria
- May need manual labeling for some episodes

## Related Resources

- LeRobot docs: https://huggingface.co/docs/lerobot
- LeRobot v3.0 format: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- LeRobot merge script: https://github.com/huggingface/lerobot/pull/924
- DAgger algorithm: https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf
- Dataset versioning: https://labelyourdata.com/articles/machine-learning/data-versioning

## Open Questions

1. Should we store raw episodes long-term or delete after conversion?
2. What is the optimal episode batch size before converting (memory constraints)?
3. Should we use HuggingFace Hub for all datasets or keep local?
4. How to handle dataset versioning - DVC, git-lfs, or manual?
5. Do we need automated quality scoring or manual review sufficient?

## Next Steps

1. Review this design with team
2. Implement Phase 1 (teleoperation data saving)
3. Test on small pilot dataset (10 episodes)
4. Iterate based on practical experience
5. Scale to full data collection pipeline
