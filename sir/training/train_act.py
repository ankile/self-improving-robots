#!/usr/bin/env python3
"""
Simple training script for ACT (Action Chunking Transformer) policy using LeRobot.

This script:
1. Loads one or more LeRobotDatasets from local storage
2. Automatically filters DAgger datasets to include only successful human corrections
3. Combines multiple datasets on-the-fly for training
4. Creates an ACT policy
5. Trains with supervised behavioral cloning (BC) loss
6. Evaluates in the environment at regular intervals

Usage (single dataset from HuggingFace Hub):
    python -m sir.training.train_act \
        --repo-ids ankile/square-v1

Usage (multiple datasets - BC + DAgger from HuggingFace Hub):
    python -m sir.training.train_act \
        --repo-ids "ankile/square-v1,ankile/square-dagger-v1"

Usage (with local datasets):
    python -m sir.training.train_act \
        --repo-ids "ankile/square-v1,ankile/square-dagger-v1" \
        --root ./data

Example with custom parameters:
    python -m sir.training.train_act \
        --repo-ids "ankile/square-v1,ankile/square-dagger-v1" \
        --batch-size 32 \
        --lr 1e-5 \
        --training-steps 20000 \
        --eval-freq 2000

Example with Weights & Biases logging and video saving:
    python -m sir.training.train_act \
        --repo-ids "ankile/square-v1,ankile/square-dagger-v1" \
        --use-wandb \
        --wandb-project act-dagger-training \
        --wandb-run-name square_bc_plus_dagger_v1 \
        --save-video

Note:
- Datasets are automatically downloaded from HuggingFace Hub if --root is not specified.
- Camera observations are automatically detected from the dataset/policy.
  The script will configure the evaluation environment to match.
- DAgger datasets (with 'source' and 'success' columns) are automatically filtered
  to include only successful human corrections (source=1, success=1).
- All dataset metadata (repo IDs, episodes, commit hashes) is logged to W&B.
"""

import argparse
import gc
import os
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress Pydantic warnings from library code (robosuite/lerobot dependencies)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)

import imageio
import numpy as np

# Robosuite imports
import robosuite as suite
import robosuite.macros as macros
import torch
import wandb
from lerobot.configs.types import FeatureType

# LeRobot imports
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

# SIR training utilities
from sir.training.wandb_artifacts import create_artifact_metadata, upload_checkpoint_to_wandb

# Set image convention to OpenCV (must be before env creation)
macros.IMAGE_CONVENTION = "opencv"

# Set MuJoCo GL backend to CGL on macOS for proper offscreen rendering
if platform.system() == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT policy with LeRobot")

    # Dataset args
    parser.add_argument(
        "--repo-ids",
        type=str,
        required=True,
        help="Comma-separated list of dataset repo IDs (e.g., 'ankile/square-v1,ankile/square-dagger-v1'). "
        "DAgger datasets (with 'source' and 'success' columns) will be automatically filtered to include only successful human corrections.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing datasets. If not specified, downloads from HuggingFace Hub to default cache directory.",
    )

    # Environment args
    parser.add_argument(
        "--env", type=str, default="Lift", help="Robosuite environment name (default: Lift)"
    )
    parser.add_argument("--robot", type=str, default="Panda", help="Robot name (default: Panda)")

    # Training args
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5, ACT default)"
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10_000,
        help="Number of training steps (default: 10000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Evaluate every N steps (default: 1000, use 10 for quick debugging)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of episodes to run during evaluation"
    )
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Maximum steps per episode during evaluation"
    )

    # ACT policy args
    parser.add_argument(
        "--chunk-size", type=int, default=20, help="Action chunk size for ACT (default: 20)"
    )
    parser.add_argument(
        "--n-obs-steps", type=int, default=1, help="Number of observation steps (default: 1)"
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=20,
        help="Number of action steps to execute per chunk (default: 20)",
    )

    # System args
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        help="Device to use (cuda/mps/cpu)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints"
    )

    # Video saving args
    parser.add_argument(
        "--save-video", action="store_true", help="Save videos of evaluation rollouts"
    )
    parser.add_argument(
        "--video-dir", type=str, default="./videos", help="Directory to save evaluation videos"
    )
    parser.add_argument(
        "--visual-aids",
        action="store_true",
        help="Enable visual aids (indicators) in evaluation environment. Disabled by default for cleaner observations.",
    )

    # Wandb args
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project", type=str, default="act-training", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Wandb entity/team name (optional)"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name (optional)"
    )

    return parser.parse_args()


class RobosuiteRenderWrapper:
    """Wrapper to add render capability to robosuite environments."""

    def __init__(self, env, render_size=(240, 320), render_camera="agentview"):
        """
        Args:
            env: Robosuite environment
            render_size: (height, width) for video rendering
            render_camera: Camera name to use for rendering videos
        """
        self.env = env
        self.render_size = render_size
        self.render_camera = render_camera

    def render(self):
        """Return an RGB frame (H, W, 3, uint8) for video recording."""
        frame = self.env.sim.render(
            camera_name=self.render_camera,
            height=self.render_size[0],
            width=self.render_size[1],
        )[
            ::-1
        ]  # Flip vertically (OpenGL convention)
        return frame

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self.env, name)


def create_robosuite_env(
    env_name,
    robot_name,
    camera_names=None,
    camera_height=None,
    camera_width=None,
    render_size=(240, 320),
    visual_aids=False,
):
    """Create a Robosuite environment for evaluation.

    Args:
        env_name: Name of the Robosuite environment
        robot_name: Name of the robot
        camera_names: List of camera names to use (None = no cameras, state-only)
        camera_height: Height of camera images (must match training data)
        camera_width: Width of camera images (must match training data)
        render_size: (height, width) for video rendering (default: 240x320)
        visual_aids: Enable visual aids/indicators in the environment (default: False)
    """
    controller_config = load_composite_controller_config(
        controller=None,  # Use default
        robot=robot_name,
    )

    # Configure camera observations based on camera_names
    use_cameras = camera_names is not None and len(camera_names) > 0

    env_config = {
        "env_name": env_name,
        "robots": robot_name,
        "controller_configs": controller_config,
        "has_renderer": False,  # No onscreen rendering during training
        "has_offscreen_renderer": use_cameras,  # Enable if cameras specified
        "ignore_done": True,
        "use_camera_obs": use_cameras,  # Enable if cameras specified
        "reward_shaping": False,  # Use sparse task success rewards only
        "control_freq": 20,
        "hard_reset": False,
    }

    # Add camera-specific config if cameras are used
    if use_cameras:
        if camera_height is None or camera_width is None:
            raise ValueError("camera_height and camera_width must be specified when using cameras")

        env_config["render_camera"] = camera_names[0]  # Default render camera
        env_config["camera_names"] = camera_names
        env_config["camera_heights"] = camera_height
        env_config["camera_widths"] = camera_width

    env = suite.make(**env_config)

    # Optionally wrap with visualization aids (disabled by default for cleaner observations)
    if visual_aids:
        env = VisualizationWrapper(env, indicator_configs=None)
        print("Visual aids enabled in evaluation environment")
    else:
        print("Visual aids disabled in evaluation environment (use --visual-aids to enable)")

    # Wrap with render capability
    render_camera = camera_names[0] if camera_names else "agentview"
    env = RobosuiteRenderWrapper(env, render_size=render_size, render_camera=render_camera)

    return env


def evaluate_policy(
    policy: ACTPolicy,
    preprocessor: Any,
    action_stats: Dict[str, Any],
    env: RobosuiteRenderWrapper,
    num_episodes: int = 10,
    max_steps: int = 400,
    device: str = "cpu",
    save_video: bool = False,
    output_dir: str = "./videos",
    step: Optional[int] = None,
) -> tuple[Dict[str, float], Optional[str]]:
    """
    Evaluate the policy in the environment.

    Args:
        policy: ACT policy
        preprocessor: Preprocessor for normalizing observations
        action_stats: Action statistics from dataset (for denormalization)
        env: Robosuite environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        device: Device to run policy on
        save_video: Whether to save rollout videos
        output_dir: Directory to save videos
        step: Current training step (for video naming)

    Returns:
        tuple: (metrics dict, video_path)
            - metrics: Dict with success_rate, avg_reward, avg_length
            - video_path: Path to saved video (None if save_video=False)
    """
    policy.eval()

    successes = []
    total_rewards = []
    episode_lengths = []

    # Video buffers for saving rollouts
    all_frames = [] if save_video else None
    episode_frames = [] if save_video else None

    # Get camera names from policy config (if any)
    camera_names = []

    if hasattr(policy.config, "image_features") and policy.config.image_features:
        # Extract camera names from image feature keys
        # e.g., "observation.images.agentview" -> "agentview"
        for img_key in policy.config.image_features:
            if img_key.startswith("observation.images."):
                cam_name = img_key.replace("observation.images.", "")
                camera_names.append(cam_name)

    # Prepare action denormalization stats (convert once for efficiency)
    # MPS doesn't support float64, so we ensure float32
    if isinstance(action_stats["mean"], torch.Tensor):
        action_mean = action_stats["mean"].to(device=device, dtype=torch.float32)
        action_std = action_stats["std"].to(device=device, dtype=torch.float32)
    else:
        action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.float32)
        action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.float32)

    for ep in range(num_episodes):
        obs = env.reset()
        policy.reset()  # Reset action queue
        episode_reward = 0
        episode_length = 0

        # Extract state observation
        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        state_keys = [k for k in state_keys if k in obs]
        state = np.concatenate([obs[k].flatten() for k in state_keys])

        for env_step in range(max_steps):
            # Prepare observation batch for policy
            obs_dict = {}

            # Add state observation
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            obs_dict["observation.state"] = state_tensor

            # Add camera observations
            for cam_name in camera_names:
                img_key = f"{cam_name}_image"
                if img_key in obs:
                    # Get image (shape: H x W x C, values 0-255)
                    # Resolution matches training data exactly (configured in env)
                    img = obs[img_key]

                    # Convert to torch tensor and normalize to [0, 1]
                    img_tensor = torch.from_numpy(img).float() / 255.0

                    # Rearrange to C x H x W format
                    img_tensor = img_tensor.permute(2, 0, 1)

                    # Add batch dimension
                    img_tensor = img_tensor.unsqueeze(0).to(device)

                    # Add to obs_dict
                    obs_dict[f"observation.images.{cam_name}"] = img_tensor

            # Apply preprocessing (normalization) to observations
            obs_dict = preprocessor(obs_dict)

            with torch.no_grad():
                # ACT's select_action returns a single action (manages action queue internally)
                action_tensor = policy.select_action(obs_dict)

                # Denormalize action manually using dataset stats
                # normalized = (value - mean) / std  ->  value = normalized * std + mean
                action_denorm = action_tensor * action_std + action_mean
                action = action_denorm.cpu().numpy()[0]

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Collect frames for video
            if save_video and episode_frames is not None:
                frame = env.render()
                episode_frames.append(frame)

            # Update state
            state = np.concatenate([obs[k].flatten() for k in state_keys])

            # Check for task success (sparse reward: 1.0 = success, 0.0 = failure)
            is_success = reward == 1.0
            if done or is_success:
                successes.append(float(is_success))
                break
        else:
            # Episode ended by max_steps
            successes.append(0.0)

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Add episode frames to all_frames and reset episode buffer
        if save_video and episode_frames is not None and all_frames is not None:
            all_frames.extend(episode_frames)
            episode_frames = []

    policy.train()

    metrics = {
        "success_rate": np.mean(successes) * 100,
        "avg_reward": np.mean(total_rewards),
        "avg_length": np.mean(episode_lengths),
    }

    # Save video if requested
    video_path = None
    if save_video and all_frames is not None and len(all_frames) > 0:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create video filename
        step_str = f"step_{step}" if step is not None else "final"
        video_name = f"eval_{step_str}.mp4"
        video_path = str(output_path / video_name)

        # Write video using imageio (use context manager for proper cleanup)
        fps = 20  # Robosuite control frequency
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in all_frames:
                writer.append_data(frame)

        print(f"✓ Saved evaluation video to: {video_path}")

    return metrics, video_path


def filter_dagger_episodes(repo_id: str, root: Path | None) -> list[int] | None:
    """
    Filter a DAgger dataset to include only successful human corrections.

    Returns a list of episode indices where source=1 (human) AND success=1 (successful),
    or None if this is not a DAgger dataset (no 'source' or 'success' columns).

    Args:
        repo_id: Dataset repository ID
        root: Root directory containing datasets

    Returns:
        List of episode indices to include, or None if not a DAgger dataset
    """
    # Load temporary dataset to check features
    temp_dataset = LeRobotDataset(repo_id, root=root)

    # Check if this is a DAgger dataset (has 'source' and 'success' columns)
    has_source = "source" in temp_dataset.features
    has_success = "success" in temp_dataset.features

    if not (has_source and has_success):
        # Not a DAgger dataset, use all episodes
        print(f"  Dataset '{repo_id}' is not a DAgger dataset (no source/success columns)")
        print(f"  Using all {temp_dataset.meta.total_episodes} episodes")
        return None

    print(
        f"  Dataset '{repo_id}' is a DAgger dataset, filtering for successful human corrections..."
    )

    # Iterate through episodes and filter
    successful_human_episodes = []

    for ep_idx in range(temp_dataset.meta.total_episodes):
        # Get episode data
        ep_data = temp_dataset.meta.episodes[ep_idx]
        from_idx = ep_data["dataset_from_index"]

        # Sample one frame from this episode to check source and success
        # (all frames in an episode should have the same source and success)
        frame = temp_dataset.hf_dataset[int(from_idx)]

        # Handle both scalar tensors and arrays
        source_val = frame["source"]
        success_val = frame["success"]

        # Convert to int - handle both tensor scalars and arrays
        if hasattr(source_val, "item"):
            source = int(source_val.item())
        elif hasattr(source_val, "__getitem__"):
            source = int(source_val[0])
        else:
            source = int(source_val)

        if hasattr(success_val, "item"):
            success = int(success_val.item())
        elif hasattr(success_val, "__getitem__"):
            success = int(success_val[0])
        else:
            success = int(success_val)

        # Keep only successful human corrections (source=1: human, success=1: successful)
        if source == 1 and success == 1:
            successful_human_episodes.append(ep_idx)

    print(
        f"  Filtered {temp_dataset.meta.total_episodes} episodes → {len(successful_human_episodes)} successful human corrections"
    )

    if len(successful_human_episodes) == 0:
        raise ValueError(
            f"DAgger dataset '{repo_id}' has no successful human corrections! "
            "Please collect some successful human demonstrations before training."
        )

    return successful_human_episodes


def train(args):
    """Main training loop."""
    # Parse comma-separated repo IDs
    repo_ids = [rid.strip() for rid in args.repo_ids.split(",")]

    print("=" * 60)
    print("Training ACT Policy with LeRobot")
    print("=" * 60)
    print(f"Datasets: {', '.join(repo_ids)}")
    print(f"Root: {args.root if args.root else 'HuggingFace Hub (default cache)'}")
    print(f"Environment: {args.env} ({args.robot})")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training steps: {args.training_steps}")
    print(f"Action chunk size: {args.chunk_size}")
    print("=" * 60)
    print()

    # Filter datasets for successful human corrections
    print("Filtering datasets...")
    episodes_dict = {}
    dataset_info = {}  # Track dataset metadata for W&B logging

    for repo_id in repo_ids:
        print(f"Processing dataset: {repo_id}")
        filtered_episodes = filter_dagger_episodes(repo_id, root=args.root)
        # Store episodes for ALL datasets (None means use all episodes)
        episodes_dict[repo_id] = filtered_episodes

        # Collect metadata for logging
        temp_meta = LeRobotDatasetMetadata(repo_id, root=args.root)
        num_episodes = (
            len(filtered_episodes) if filtered_episodes is not None else temp_meta.total_episodes
        )
        dataset_info[repo_id] = {
            "total_episodes": temp_meta.total_episodes,
            "used_episodes": num_episodes,
            "total_frames": temp_meta.total_frames,
            "revision": temp_meta.revision,
        }
        print()

    # Initialize Weights & Biases with dataset metadata
    config = {
        # Dataset config
        "dataset/repo_ids": repo_ids,
        "dataset/root": args.root,
        # Environment config
        "environment/env_name": args.env,
        "environment/robot": args.robot,
        # Training config
        "training/batch_size": args.batch_size,
        "training/lr": args.lr,
        "training/training_steps": args.training_steps,
        "training/eval_freq": args.eval_freq,
        "training/eval_episodes": args.eval_episodes,
        "training/max_steps": args.max_steps,
        # Policy config
        "policy/chunk_size": args.chunk_size,
        "policy/n_obs_steps": args.n_obs_steps,
        "policy/n_action_steps": args.n_action_steps,
        # System config
        "system/device": args.device,
        "system/num_workers": args.num_workers,
        "system/checkpoint_dir": args.checkpoint_dir,
    }

    # Add dataset-specific metadata
    for repo_id, info in dataset_info.items():
        safe_name = repo_id.replace("/", "_")
        config[f"dataset/{safe_name}/total_episodes"] = info["total_episodes"]
        config[f"dataset/{safe_name}/used_episodes"] = info["used_episodes"]
        config[f"dataset/{safe_name}/total_frames"] = info["total_frames"]
        config[f"dataset/{safe_name}/revision"] = info["revision"]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
        mode="online" if args.use_wandb else "disabled",
    )
    print(f"✓ Initialized Weights & Biases (project: {args.wandb_project})")
    print()

    # Load dataset metadata first to get features and stats (from first dataset)
    print("Loading dataset metadata...")
    primary_repo_id = repo_ids[0]
    dataset_metadata = LeRobotDatasetMetadata(primary_repo_id, root=args.root)
    fps = dataset_metadata.fps
    print(f"✓ Dataset metadata loaded")
    print(f"  FPS: {fps}")
    print()

    # Convert dataset features to policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print("Input features:")
    for key, ft in input_features.items():
        print(f"  {key}: shape={ft.shape}, type={ft.type}")
    print("Output features:")
    for key, ft in output_features.items():
        print(f"  {key}: shape={ft.shape}, type={ft.type}")
    print()

    # Create ACT policy configuration
    # Note: ACT automatically detects image features from input_features based on FeatureType.VISUAL
    print("Creating ACT policy...")
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.lr,
    )

    # Create policy
    policy = ACTPolicy(config)
    policy.train()
    policy.to(args.device)
    print(f"✓ ACT policy created")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()

    # Create pre/post processors for normalization
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=dataset_metadata.stats
    )

    # Setup delta timestamps for temporal data
    # ACT default: single observation step, multiple action steps
    delta_timestamps = {
        "action": [i / fps for i in range(args.chunk_size)],  # Chunk of future actions
    }

    # Add observation keys based on what's in the dataset
    for key in input_features.keys():
        if key.startswith("observation."):
            delta_timestamps[key] = [0.0]  # Current observation only

    print("Delta timestamps configuration:")
    for key, timestamps in delta_timestamps.items():
        if len(timestamps) <= 3:
            print(f"  {key}: {timestamps}")
        else:
            print(
                f"  {key}: [{timestamps[0]}, {timestamps[1]}, ..., {timestamps[-1]}] ({len(timestamps)} steps)"
            )
    print()

    # Load dataset(s) with delta timestamps
    print("Loading dataset(s)...")
    dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        root=args.root,
        delta_timestamps=delta_timestamps,
        episodes=episodes_dict,  # Dict with entry for each repo_id (None = use all)
    )
    print("✓ Datasets loaded:")
    for repo_id, info in dataset_info.items():
        print(f"  {repo_id}: {info['used_episodes']} episodes (of {info['total_episodes']} total)")
    print(f"  Total frames: {len(dataset)}")
    print()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=True,  # Drop last incomplete batch
    )

    # Create optimizer (use policy's get_optim_params for proper param groups)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr)

    # Detect camera names and resolution from dataset features
    camera_names = []
    camera_height = None
    camera_width = None

    # Check if we have image features in the dataset
    for key, feature in features.items():
        if key.startswith("observation.images.") and feature.type == FeatureType.VISUAL:
            cam_name = key.replace("observation.images.", "")
            camera_names.append(cam_name)

            # Extract image shape (C, H, W)
            if len(feature.shape) == 3:
                c, h, w = feature.shape
                # All cameras should have the same resolution
                if camera_height is None:
                    camera_height = h
                    camera_width = w
                elif camera_height != h or camera_width != w:
                    print(
                        f"WARNING: Camera {cam_name} has different resolution ({h}x{w}) than expected ({camera_height}x{camera_width})"
                    )

    if camera_names:
        print(f"Detected camera features from dataset: {camera_names}")
        print(f"Camera resolution: {camera_height}×{camera_width}")
    else:
        print("No camera features detected in dataset (state-only policy)")

    # Create environment for evaluation
    print("Creating evaluation environment...")
    eval_env = create_robosuite_env(
        args.env,
        args.robot,
        camera_names=camera_names if camera_names else None,
        camera_height=camera_height,
        camera_width=camera_width,
        visual_aids=args.visual_aids,
    )
    print("✓ Evaluation environment created")
    print()

    # Create checkpoint directory
    # Use combined dataset name for checkpoint directory
    combined_name = "+".join([rid.split("/")[-1] for rid in repo_ids])
    checkpoint_dir = Path(args.checkpoint_dir) / combined_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print()

    # Training loop
    print("Starting training...")
    print("=" * 60)

    best_success_rate = 0.0
    step = 0
    done = False

    while not done:
        for batch in dataloader:
            policy.train()

            # Preprocess batch (normalization)
            batch = preprocessor(batch)

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(args.device)

            # Remove temporal dimension from observations (n_obs_steps=1)
            # ACT expects observations without temporal dimension when using single observation step
            for key in list(batch.keys()):
                if key.startswith("observation.") and not key.endswith("_is_pad"):
                    if isinstance(batch[key], torch.Tensor) and batch[key].shape[1] == 1:
                        # Squeeze out the temporal dimension (dim=1)
                        batch[key] = batch[key].squeeze(1)

            # Forward pass (returns loss and loss dict)
            loss, loss_dict = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if step % 100 == 0:
                loss_str = f"step: {step}/{args.training_steps} loss: {loss.item():.4f}"
                if "l1_loss" in loss_dict:
                    loss_str += f" l1: {loss_dict['l1_loss']:.4f}"
                if "kl_loss" in loss_dict:
                    loss_str += f" kl: {loss_dict['kl_loss']:.4f}"
                print(loss_str)

                # Log to wandb
                if args.use_wandb:
                    wandb_log = {
                        "train/loss": loss.item(),
                        "train/step": step,
                    }
                    if "l1_loss" in loss_dict:
                        wandb_log["train/l1_loss"] = loss_dict["l1_loss"]
                    if "kl_loss" in loss_dict:
                        wandb_log["train/kl_loss"] = loss_dict["kl_loss"]
                    wandb.log(wandb_log, step=step)

            # Evaluation
            if step > 0 and step % args.eval_freq == 0:
                print("\nEvaluating policy...")
                metrics, video_path = evaluate_policy(
                    policy,
                    preprocessor,
                    dataset_metadata.stats["action"],
                    eval_env,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    device=args.device,
                    save_video=args.save_video,
                    output_dir=args.video_dir,
                    step=step,
                )

                print(f"Evaluation Results (step {step}):")
                print(f"  Success Rate: {metrics['success_rate']:.1f}%")
                print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
                print(f"  Avg Length: {metrics['avg_length']:.1f}")
                print()

                # Update best success rate
                is_best = False
                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                    is_best = True

                # Log to wandb
                if args.use_wandb:
                    wandb_log = {
                        "eval/success_rate": metrics["success_rate"],
                        "eval/avg_reward": metrics["avg_reward"],
                        "eval/avg_length": metrics["avg_length"],
                        "eval/best_success_rate": best_success_rate,
                    }
                    # Log video if available
                    if video_path is not None:
                        wandb_log["eval/video"] = wandb.Video(video_path, format="mp4")
                    wandb.log(wandb_log, step=step)

                    # Force garbage collection to clean up video files
                    if video_path is not None:
                        gc.collect()

                # Save best model
                if is_best:
                    checkpoint_path = checkpoint_dir / "best_model"
                    policy.save_pretrained(checkpoint_path)
                    preprocessor.save_pretrained(checkpoint_path)
                    postprocessor.save_pretrained(checkpoint_path)
                    print(f"✓ Saved best model (success rate: {best_success_rate:.1f}%)")

                    # Upload to W&B for version control and DAgger preparation
                    if args.use_wandb:
                        metadata = create_artifact_metadata(
                            repo_id=combined_name,
                            success_rate=best_success_rate / 100.0,
                            avg_reward=metrics["avg_reward"],
                            step=step,
                            is_best=True,
                            dataset_size=len(dataset),
                        )
                        # Add detailed dataset info to metadata
                        for i, repo_id in enumerate(repo_ids):
                            metadata[f"dataset_{i}_repo_id"] = repo_id
                            metadata[f"dataset_{i}_episodes"] = dataset_info[repo_id][
                                "used_episodes"
                            ]
                            metadata[f"dataset_{i}_revision"] = dataset_info[repo_id]["revision"]

                        artifact_name = f"act-{combined_name}-best-step-{step}"
                        upload_checkpoint_to_wandb(
                            checkpoint_path=checkpoint_path,
                            artifact_name=artifact_name,
                            description=f"Best ACT policy trained on {', '.join(repo_ids)} (success: {best_success_rate:.1f}%, step: {step})",
                            metadata=metadata,
                        )
                    print()

            # Save periodic checkpoint
            if step > 0 and step % 5000 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_path)
                preprocessor.save_pretrained(checkpoint_path)
                postprocessor.save_pretrained(checkpoint_path)
                print(f"✓ Saved checkpoint at step {step}")

                # Upload to W&B for version control
                if args.use_wandb:
                    artifact_name = f"act-{combined_name}-checkpoint-step-{step}"
                    checkpoint_metadata = {"step": step, "combined_name": combined_name}
                    # Add dataset info
                    for i, repo_id in enumerate(repo_ids):
                        checkpoint_metadata[f"dataset_{i}_repo_id"] = repo_id
                        checkpoint_metadata[f"dataset_{i}_episodes"] = dataset_info[repo_id][
                            "used_episodes"
                        ]
                        checkpoint_metadata[f"dataset_{i}_revision"] = dataset_info[repo_id][
                            "revision"
                        ]

                    upload_checkpoint_to_wandb(
                        checkpoint_path=checkpoint_path,
                        artifact_name=artifact_name,
                        description=f"ACT policy checkpoint at step {step}",
                        metadata=checkpoint_metadata,
                    )
                print()

            step += 1
            if step >= args.training_steps:
                done = True
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete! Running final evaluation...")
    print("=" * 60)
    metrics, video_path = evaluate_policy(
        policy,
        preprocessor,
        dataset_metadata.stats["action"],
        eval_env,
        num_episodes=args.eval_episodes * 2,  # More episodes for final eval
        max_steps=args.max_steps,
        device=args.device,
        save_video=args.save_video,
        output_dir=args.video_dir,
        step=None,  # Final eval (no step number)
    )

    print("\nFinal Evaluation Results:")
    print(f"  Success Rate: {metrics['success_rate']:.1f}%")
    print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"  Avg Length: {metrics['avg_length']:.1f}")
    print()

    # Log final evaluation to wandb
    if args.use_wandb:
        wandb_log = {
            "final_eval/success_rate": metrics["success_rate"],
            "final_eval/avg_reward": metrics["avg_reward"],
            "final_eval/avg_length": metrics["avg_length"],
        }
        # Log video if available
        if video_path is not None:
            wandb_log["final_eval/video"] = wandb.Video(video_path, format="mp4")
        wandb.log(wandb_log, step=step)

        # Force garbage collection to clean up video files
        if video_path is not None:
            gc.collect()

    # Save final model
    checkpoint_path = checkpoint_dir / "final_model"
    policy.save_pretrained(checkpoint_path)
    preprocessor.save_pretrained(checkpoint_path)
    postprocessor.save_pretrained(checkpoint_path)
    print(f"✓ Saved final model to {checkpoint_path}")

    # Upload final model to W&B
    if args.use_wandb:
        metadata = create_artifact_metadata(
            repo_id=combined_name,
            success_rate=metrics["success_rate"] / 100.0,
            avg_reward=metrics["avg_reward"],
            step=step,
            is_best=False,
            dataset_size=len(dataset),
        )
        # Add detailed dataset info to metadata
        for i, repo_id in enumerate(repo_ids):
            metadata[f"dataset_{i}_repo_id"] = repo_id
            metadata[f"dataset_{i}_episodes"] = dataset_info[repo_id]["used_episodes"]
            metadata[f"dataset_{i}_revision"] = dataset_info[repo_id]["revision"]

        artifact_name = f"act-{combined_name}-final"
        upload_checkpoint_to_wandb(
            checkpoint_path=checkpoint_path,
            artifact_name=artifact_name,
            description=f"Final ACT policy trained on {', '.join(repo_ids)} after {step} steps (success: {metrics['success_rate']:.1f}%)",
            metadata=metadata,
        )

    # Cleanup
    eval_env.close()

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        print("✓ Finished Weights & Biases logging")

    print("\nDone!")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
