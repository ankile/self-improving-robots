#!/usr/bin/env python3
"""
Simple SpaceMouse teleoperation for ManiSkill environments.

This script allows you to control a robot in ManiSkill using a SpaceMouse device.
The SpaceMouse provides 6-DOF control (3D position + 3D rotation) for the end-effector.

The default control mode is pd_ee_target_delta_pose, which uses target-based control.
This means the robot maintains its commanded position even when holding objects or under
external forces, preventing drift and drooping that occurs with regular delta control.

Usage:
    python -m sir.teleoperation.spacemouse_teleop -e PegInsertionSide-v1

Controls:
    - SpaceMouse translation: Move robot end-effector in XYZ
    - SpaceMouse rotation: Rotate robot end-effector
    - Left button: Toggle gripper open/close
    - '1' key: Mark episode as success and reset
    - '0' key: Mark episode as failure and reset
    - Ctrl+C: Quit
"""

import argparse
import gymnasium as gym
import numpy as np
import pyspacemouse
import mani_skill.envs  # Register ManiSkill environments
from pathlib import Path
from datetime import datetime

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes

from .utils import apply_deadzone, parse_axis_mapping, KeyboardListener
from .robot_config import create_custom_panda


def parse_args():
    parser = argparse.ArgumentParser(description="SpaceMouse teleoperation for ManiSkill")
    parser.add_argument(
        "-e", "--env-id",
        type=str,
        default="PegInsertionSide-v1",
        help="Environment ID (e.g., PegInsertionSide-v1, PickCube-v1, StackCube-v1)"
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_ee_target_delta_pose",
        help="Control mode for the robot. Options: pd_ee_target_delta_pose, pd_ee_delta_pose, pd_ee_target_delta_pos, pd_ee_delta_pos"
    )
    parser.add_argument(
        "--robot-uid",
        type=str,
        default="panda",
        help="Robot to use (default: panda)"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode (default: 10000, set to 0 for unlimited)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Speed multiplier for SpaceMouse control (default: 0.15)"
    )
    parser.add_argument(
        "--rot-speed",
        type=float,
        default=0.3,
        help="Rotation speed multiplier for SpaceMouse control (default: 0.3)"
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=2000.0,
        help="PD controller stiffness (default: 2000.0, higher = more responsive)"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=200.0,
        help="PD controller damping (default: 200.0, higher = less oscillation)"
    )
    parser.add_argument(
        "--force-limit",
        type=float,
        default=200.0,
        help="Maximum force for PD controller (default: 200.0)"
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.02,
        help="SpaceMouse deadzone threshold (default: 0.02, values below this are ignored)"
    )
    parser.add_argument(
        "--axis-mapping",
        type=str,
        default="-yxz",
        help="SpaceMouse axis mapping to robot frame (default: -yxz). Use permutation like 'yxz' or '-x-yz' to remap axes"
    )
    parser.add_argument(
        "--rot-axis-mapping",
        type=str,
        default="-x-yz",
        help="SpaceMouse rotation axis mapping (default: -x-yz). Use permutation like 'yxz' or '-x-yz' to remap rotation axes"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save teleoperation data to LeRobotDataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to save dataset (default: ./data)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset. If not specified, will be generated as {env_id}_{timestamp}"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SpaceMouse Teleoperation for ManiSkill")
    print("=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Control mode: {args.control_mode}")
    print(f"Robot: {args.robot_uid}")
    print(f"Translation speed: {args.speed}")
    print(f"Rotation speed: {args.rot_speed}")
    print(f"PD Controller: stiffness={args.stiffness}, damping={args.damping}, force_limit={args.force_limit}")
    print(f"Deadzone: {args.deadzone}")
    print(f"Axis mapping: {args.axis_mapping} (translation), {args.rot_axis_mapping} (rotation)")

    # Data saving setup
    dataset = None
    if args.save_data:
        dataset_name = args.dataset_name
        if dataset_name is None:
            # Generate dataset name from env_id and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = args.env_id.replace("-", "_").lower()
            dataset_name = f"{env_name}_{timestamp}"

        dataset_path = Path(args.dataset_path) / dataset_name
        print(f"Data saving: ENABLED")
        print(f"Dataset path: {dataset_path}")
        print(f"Dataset name: {dataset_name}")
    else:
        print(f"Data saving: DISABLED")
    print()

    # Parse axis mappings
    try:
        axis_remap = parse_axis_mapping(args.axis_mapping)
        rot_axis_remap = parse_axis_mapping(args.rot_axis_mapping)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Initialize SpaceMouse
    print("Initializing SpaceMouse...")
    success = pyspacemouse.open()
    if not success:
        print("ERROR: Could not open SpaceMouse device!")
        print("Make sure your SpaceMouse is plugged in and you have the necessary permissions.")
        return
    print("✓ SpaceMouse connected successfully!")
    print()

    # Create custom robot with tuned control parameters
    print("Configuring robot controller...")
    if args.robot_uid == "panda":
        custom_robot_cls = create_custom_panda(args.stiffness, args.damping, args.force_limit)
        robot_uid = "panda_custom_teleop"
    else:
        robot_uid = args.robot_uid
    print(f"✓ Robot configured with PD gains!")
    print()

    # Create environment
    print("Creating environment...")
    env_kwargs = {
        "obs_mode": "state",  # Use state observations
        "control_mode": args.control_mode,
        "render_mode": "human",  # Use human mode so viewer is managed during reconfiguration
        "reward_mode": "dense",
        "enable_shadow": True,
        "robot_uids": robot_uid,
    }

    # Set max episode steps (0 means unlimited, otherwise use specified value)
    if args.max_episode_steps == 0:
        env_kwargs["max_episode_steps"] = 1000000  # Effectively unlimited
    else:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **env_kwargs)

    print("✓ Environment created!")
    print()

    # Print control instructions
    print("=" * 60)
    print("CONTROLS:")
    print("=" * 60)
    print("SpaceMouse translation: Move end-effector in XYZ")
    print("SpaceMouse rotation: Rotate end-effector")
    print("Left button: Toggle gripper open/close")
    print("'1' key: Mark episode as SUCCESS and reset")
    print("'0' key: Mark episode as FAILURE and reset")
    print("Ctrl+C: Quit")
    print("=" * 60)
    print()

    # Action space info
    if "pose" in args.control_mode:
        action_dim = 7  # 3 pos + 3 rot + 1 gripper
        print(f"Action space: 7D (3 translation + 3 rotation + 1 gripper)")
    elif "pos" in args.control_mode:
        action_dim = 4  # 3 pos + 1 gripper
        print(f"Action space: 4D (3 translation + 1 gripper)")
    else:
        action_dim = env.action_space.shape[0]
        print(f"Action space: {action_dim}D")
    print()

    # Initialize keyboard listener
    kbd_listener = KeyboardListener()
    print("✓ Keyboard listener initialized (works regardless of window focus)")

    # Teleoperation loop
    gripper_open = True
    prev_button_left = False
    episode_count = 0
    saved_episode_count = 0

    # Episode buffer for data collection
    episode_buffer = None
    if args.save_data:
        episode_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    obs, info = env.reset(seed=0)
    print(f"Environment reset (Episode {episode_count})")
    print(f"Task info: {info}")
    print()

    # Store initial observation if saving data
    if args.save_data:
        episode_buffer["observations"].append(obs.copy())

    try:
        while True:
            # Render the SAPIEN viewer (must be called every iteration)
            env.render_human()

            # Read SpaceMouse state
            state = pyspacemouse.read()

            # Apply deadzone to filter out noise
            x = apply_deadzone(state.x, args.deadzone)
            y = apply_deadzone(state.y, args.deadzone)
            z = apply_deadzone(state.z, args.deadzone)
            roll = apply_deadzone(state.roll, args.deadzone)
            pitch = apply_deadzone(state.pitch, args.deadzone)
            yaw = apply_deadzone(state.yaw, args.deadzone)

            # Remap axes according to user-specified mapping
            x, y, z = axis_remap(x, y, z)
            roll, pitch, yaw = rot_axis_remap(roll, pitch, yaw)

            # Build action from SpaceMouse input
            if "pose" in args.control_mode:
                # 6-DOF control: translation (xyz) + rotation (rpy)
                action = np.array([
                    x * args.speed,      # x translation
                    y * args.speed,      # y translation
                    z * args.speed,      # z translation
                    roll * args.rot_speed,   # roll rotation
                    pitch * args.rot_speed,  # pitch rotation
                    yaw * args.rot_speed,    # yaw rotation
                    1.0 if gripper_open else -1.0  # gripper (1=open, -1=close)
                ], dtype=np.float32)
            elif "pos" in args.control_mode:
                # 3-DOF control: translation only (xyz)
                action = np.array([
                    x * args.speed,      # x translation
                    y * args.speed,      # y translation
                    z * args.speed,      # z translation
                    1.0 if gripper_open else -1.0  # gripper
                ], dtype=np.float32)
            else:
                # Fallback: zero action
                action = np.zeros(action_dim, dtype=np.float32)

            # Handle buttons
            button_left = len(state.buttons) > 0 and state.buttons[0] == 1

            # Left button: toggle gripper (on button press, not hold)
            if button_left and not prev_button_left:
                gripper_open = not gripper_open
                print(f"Gripper: {'OPEN' if gripper_open else 'CLOSED'}")

            prev_button_left = button_left

            # Check for keyboard input
            key = kbd_listener.read_key()
            should_reset = False
            episode_success = None

            if key == '1':
                print()
                print("=" * 60)
                print("SUCCESS! Resetting environment...")
                print("=" * 60)
                print()
                should_reset = True
                episode_success = True
            elif key == '0':
                print()
                print("=" * 60)
                print("FAILURE! Resetting environment...")
                print("=" * 60)
                print()
                should_reset = True
                episode_success = False

            # Save episode data if requested
            if should_reset and args.save_data and len(episode_buffer["actions"]) > 0:
                # Save episode to dataset
                if dataset is None:
                    # Initialize dataset on first save
                    dataset = LeRobotDataset.create(
                        repo_id=dataset_name,
                        fps=30,  # Approximate control frequency
                        root=args.dataset_path,
                        robot_type="panda",
                        features={
                            "observation.state": {
                                "dtype": "float32",
                                "shape": (obs.shape[0],),
                                "names": ["state_dim_" + str(i) for i in range(obs.shape[0])]
                            },
                            "action": {
                                "dtype": "float32",
                                "shape": (action_dim,),
                                "names": ["action_dim_" + str(i) for i in range(action_dim)]
                            }
                        }
                    )
                    print(f"✓ Dataset initialized at {dataset_path}")

                # Convert episode buffer to arrays
                episode_data = {
                    "observation.state": np.array(episode_buffer["observations"], dtype=np.float32),
                    "action": np.array(episode_buffer["actions"], dtype=np.float32),
                    "episode_index": np.full(len(episode_buffer["actions"]), saved_episode_count, dtype=np.int64),
                    "frame_index": np.arange(len(episode_buffer["actions"]), dtype=np.int64),
                    "timestamp": np.arange(len(episode_buffer["actions"]), dtype=np.float32) / 30.0,  # Approximate
                    "next.done": np.array(episode_buffer["dones"], dtype=bool),
                    "next.reward": np.array(episode_buffer["rewards"], dtype=np.float32),
                    "next.success": np.full(len(episode_buffer["actions"]), episode_success, dtype=bool),
                }

                # Add episode to dataset
                dataset.add_episode(episode_data)
                saved_episode_count += 1
                print(f"✓ Episode saved to dataset (Total saved: {saved_episode_count})")

                # Clear episode buffer
                episode_buffer = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                }

            if should_reset:
                episode_count += 1
                obs, info = env.reset()
                print(f"Environment reset (Episode {episode_count})")
                gripper_open = True
                # Store initial observation for new episode if saving data
                if args.save_data:
                    episode_buffer["observations"].append(obs.copy())
                continue

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect data for episode buffer
            if args.save_data:
                episode_buffer["observations"].append(obs.copy())
                episode_buffer["actions"].append(action.copy())
                episode_buffer["rewards"].append(reward)
                episode_buffer["dones"].append(terminated or truncated)

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Interrupted by user!")
        print("=" * 60)

    finally:
        # Clean up
        print()
        print("Cleaning up...")
        if args.save_data and dataset is not None:
            print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
            # Dataset is automatically saved when going out of scope
        print("Stopping keyboard listener...")
        kbd_listener.close()
        print("Closing SpaceMouse...")
        pyspacemouse.close()
        print("Closing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
