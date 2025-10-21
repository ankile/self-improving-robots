#!/usr/bin/env python3
"""
Simple training script for ACT (Action Chunking Transformer) policy using LeRobot.

This script:
1. Loads a LeRobotDataset from local storage
2. Creates an ACT policy
3. Trains with supervised behavioral cloning (BC) loss
4. Evaluates in the environment at regular intervals

Usage:
    python -m sir.training.train_act \
        --repo-id lift_minimal_state \
        --root ./data/test_filtered \
        --env Lift \
        --robot Panda

Example with custom parameters:
    python -m sir.training.train_act \
        --repo-id lift_minimal_state \
        --root ./data/test_filtered \
        --env Lift \
        --robot Panda \
        --batch-size 32 \
        --lr 1e-5 \
        --training-steps 20000 \
        --eval-freq 2000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType

# Robosuite imports
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import robosuite.macros as macros

# Set image convention to OpenCV (must be before env creation)
macros.IMAGE_CONVENTION = "opencv"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT policy with LeRobot")

    # Dataset args
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Dataset repo ID (name)")
    parser.add_argument("--root", type=str, default="./data",
                        help="Root directory containing datasets")

    # Environment args
    parser.add_argument("--env", type=str, default="Lift",
                        help="Robosuite environment name")
    parser.add_argument("--robot", type=str, default="Panda",
                        help="Robot name")

    # Training args
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5, ACT default)")
    parser.add_argument("--training-steps", type=int, default=10000,
                        help="Number of training steps (default: 10000)")
    parser.add_argument("--eval-freq", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes to run during evaluation")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Maximum steps per episode during evaluation")

    # ACT policy args
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Action chunk size for ACT (default: 20)")
    parser.add_argument("--n-obs-steps", type=int, default=1,
                        help="Number of observation steps (default: 1)")
    parser.add_argument("--n-action-steps", type=int, default=20,
                        help="Number of action steps to execute per chunk (default: 20)")

    # System args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")

    return parser.parse_args()


def create_robosuite_env(env_name, robot_name):
    """Create a Robosuite environment for evaluation."""
    controller_config = load_composite_controller_config(
        controller=None,  # Use default
        robot=robot_name,
    )

    env_config = {
        "env_name": env_name,
        "robots": robot_name,
        "controller_configs": controller_config,
        "has_renderer": False,  # No onscreen rendering during training
        "has_offscreen_renderer": False,  # No camera observations for now
        "render_camera": "agentview",
        "ignore_done": True,
        "use_camera_obs": False,
        "reward_shaping": True,
        "control_freq": 20,
        "hard_reset": False,
    }

    env = suite.make(**env_config)
    env = VisualizationWrapper(env, indicator_configs=None)

    return env


def evaluate_policy(policy, postprocessor, env, num_episodes=10, max_steps=400, device="cpu"):
    """
    Evaluate the policy in the environment.

    Args:
        policy: ACT policy
        postprocessor: Postprocessor for denormalizing actions
        env: Robosuite environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        device: Device to run policy on

    Returns:
        dict: Evaluation metrics (success_rate, avg_reward, avg_length)
    """
    policy.eval()

    successes = []
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs = env.reset()
        policy.reset()  # Reset action queue
        episode_reward = 0
        episode_length = 0

        # Extract state observation
        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        state_keys = [k for k in state_keys if k in obs]
        state = np.concatenate([obs[k].flatten() for k in state_keys])

        for step in range(max_steps):
            # Prepare observation batch for policy
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            obs_dict = {"observation.state": state_tensor}

            with torch.no_grad():
                # ACT's select_action returns a single action (manages action queue internally)
                action_tensor = policy.select_action(obs_dict)

                # Denormalize action
                action_batch = {"action": action_tensor}
                action_batch = postprocessor(action_batch)
                action = action_batch["action"].cpu().numpy()[0]

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Update state
            state = np.concatenate([obs[k].flatten() for k in state_keys])

            # Check for task success
            if done or info.get("success", False):
                successes.append(float(info.get("success", False)))
                break
        else:
            # Episode ended by max_steps
            successes.append(0.0)

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    policy.train()

    metrics = {
        "success_rate": np.mean(successes) * 100,
        "avg_reward": np.mean(total_rewards),
        "avg_length": np.mean(episode_lengths),
    }

    return metrics


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("Training ACT Policy with LeRobot")
    print("=" * 60)
    print(f"Dataset: {args.repo_id}")
    print(f"Root: {args.root}")
    print(f"Environment: {args.env} ({args.robot})")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training steps: {args.training_steps}")
    print(f"Action chunk size: {args.chunk_size}")
    print("=" * 60)
    print()

    # Load dataset metadata first to get features and stats
    print("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = dataset_metadata.fps
    print(f"✓ Dataset metadata loaded")
    print(f"  Total episodes: {dataset_metadata.total_episodes}")
    print(f"  Total frames: {dataset_metadata.total_frames}")
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
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_metadata.stats)

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
            print(f"  {key}: [{timestamps[0]}, {timestamps[1]}, ..., {timestamps[-1]}] ({len(timestamps)} steps)")
    print()

    # Load dataset with delta timestamps
    print("Loading dataset...")
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
    )
    print(f"✓ Dataset loaded: {len(dataset)} frames")
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
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=args.lr)

    # Create environment for evaluation
    print("Creating evaluation environment...")
    eval_env = create_robosuite_env(args.env, args.robot)
    print("✓ Evaluation environment created")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.repo_id
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

            # Evaluation
            if step > 0 and step % args.eval_freq == 0:
                print("\nEvaluating policy...")
                metrics = evaluate_policy(
                    policy,
                    postprocessor,
                    eval_env,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    device=args.device,
                )

                print(f"Evaluation Results (step {step}):")
                print(f"  Success Rate: {metrics['success_rate']:.1f}%")
                print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
                print(f"  Avg Length: {metrics['avg_length']:.1f}")
                print()

                # Save best model
                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                    checkpoint_path = checkpoint_dir / "best_model"
                    policy.save_pretrained(checkpoint_path)
                    preprocessor.save_pretrained(checkpoint_path)
                    postprocessor.save_pretrained(checkpoint_path)
                    print(f"✓ Saved best model (success rate: {best_success_rate:.1f}%)")
                    print()

            # Save periodic checkpoint
            if step > 0 and step % 5000 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
                policy.save_pretrained(checkpoint_path)
                preprocessor.save_pretrained(checkpoint_path)
                postprocessor.save_pretrained(checkpoint_path)
                print(f"✓ Saved checkpoint at step {step}")
                print()

            step += 1
            if step >= args.training_steps:
                done = True
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete! Running final evaluation...")
    print("=" * 60)
    metrics = evaluate_policy(
        policy,
        postprocessor,
        eval_env,
        num_episodes=args.eval_episodes * 2,  # More episodes for final eval
        max_steps=args.max_steps,
        device=args.device,
    )

    print(f"\nFinal Evaluation Results:")
    print(f"  Success Rate: {metrics['success_rate']:.1f}%")
    print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"  Avg Length: {metrics['avg_length']:.1f}")
    print()

    # Save final model
    checkpoint_path = checkpoint_dir / "final_model"
    policy.save_pretrained(checkpoint_path)
    preprocessor.save_pretrained(checkpoint_path)
    postprocessor.save_pretrained(checkpoint_path)
    print(f"✓ Saved final model to {checkpoint_path}")

    # Cleanup
    eval_env.close()
    print("\nDone!")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
