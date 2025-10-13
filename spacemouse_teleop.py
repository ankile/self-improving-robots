#!/usr/bin/env python3
"""
Simple SpaceMouse teleoperation for ManiSkill environments.

This script allows you to control a robot in ManiSkill using a SpaceMouse device.
The SpaceMouse provides 6-DOF control (3D position + 3D rotation) for the end-effector.

Usage:
    python spacemouse_teleop.py -e PegInsertionSide-v1

Controls:
    - SpaceMouse translation: Move robot end-effector in XYZ
    - SpaceMouse rotation: Rotate robot end-effector
    - Left button: Toggle gripper open/close
    - Right button: Save trajectory and start new episode
    - Press Ctrl+C in terminal: Quit and save all trajectories
"""

import argparse
import gymnasium as gym
import numpy as np
import pyspacemouse
from pathlib import Path
import mani_skill.envs  # Register ManiSkill environments
from mani_skill.utils.wrappers.record import RecordEpisode


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
        default="pd_ee_delta_pose",
        help="Control mode for the robot. Options: pd_ee_delta_pose, pd_ee_delta_pos"
    )
    parser.add_argument(
        "--robot-uid",
        type=str,
        default="panda",
        help="Robot to use (default: panda)"
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="demos",
        help="Directory to save demonstration data"
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
        default=0.05,
        help="Speed multiplier for SpaceMouse control (default: 0.05)"
    )
    parser.add_argument(
        "--rot-speed",
        type=float,
        default=0.1,
        help="Rotation speed multiplier for SpaceMouse control (default: 0.1)"
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
    print()

    # Initialize SpaceMouse
    print("Initializing SpaceMouse...")
    success = pyspacemouse.open()
    if not success:
        print("ERROR: Could not open SpaceMouse device!")
        print("Make sure your SpaceMouse is plugged in and you have the necessary permissions.")
        return
    print("✓ SpaceMouse connected successfully!")
    print()

    # Create output directory
    output_dir = Path(args.record_dir) / args.env_id / "spacemouse"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving demonstrations to: {output_dir}")
    print()

    # Create environment
    print("Creating environment...")
    env_kwargs = {
        "obs_mode": "state",  # Use state observations
        "control_mode": args.control_mode,
        "render_mode": "rgb_array",
        "reward_mode": "dense",
        "enable_shadow": True,
    }

    # Only add robot_uids if specified and not default
    if args.robot_uid != "panda":
        env_kwargs["robot_uids"] = args.robot_uid

    # Set max episode steps (0 means unlimited, otherwise use specified value)
    if args.max_episode_steps == 0:
        env_kwargs["max_episode_steps"] = 1000000  # Effectively unlimited
    else:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **env_kwargs)

    # Wrap with recording
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="SpaceMouse teleoperation"
    )

    print("✓ Environment created!")
    print()

    # Print control instructions
    print("=" * 60)
    print("CONTROLS:")
    print("=" * 60)
    print("SpaceMouse translation: Move end-effector in XYZ")
    print("SpaceMouse rotation: Rotate end-effector")
    print("Left button: Toggle gripper open/close")
    print("Right button: Save trajectory and start new episode")
    print("Ctrl+C: Quit and save all trajectories")
    print("=" * 60)
    print()

    # Action space info
    if args.control_mode == "pd_ee_delta_pose":
        action_dim = 7  # 3 pos + 3 rot + 1 gripper
        print(f"Action space: 7D (3 translation + 3 rotation + 1 gripper)")
    elif args.control_mode == "pd_ee_delta_pos":
        action_dim = 4  # 3 pos + 1 gripper
        print(f"Action space: 4D (3 translation + 1 gripper)")
    else:
        action_dim = env.action_space.shape[0]
        print(f"Action space: {action_dim}D")
    print()

    # Teleoperation loop
    num_trajs = 0
    seed = 0
    gripper_open = True
    prev_button_left = False
    prev_button_right = False

    obs, info = env.reset(seed=seed)
    print(f"Starting trajectory {num_trajs + 1} (seed={seed})")
    print(f"Task info: {info}")
    print()

    try:
        while True:
            # Open viewer window
            env.render_human()

            # Read SpaceMouse state
            state = pyspacemouse.read()

            # Build action from SpaceMouse input
            if args.control_mode == "pd_ee_delta_pose":
                # 6-DOF control: translation (xyz) + rotation (rpy)
                action = np.array([
                    state.x * args.speed,      # x translation
                    state.y * args.speed,      # y translation
                    state.z * args.speed,      # z translation
                    state.roll * args.rot_speed,   # roll rotation
                    state.pitch * args.rot_speed,  # pitch rotation
                    state.yaw * args.rot_speed,    # yaw rotation
                    1.0 if gripper_open else -1.0  # gripper (1=open, -1=close)
                ], dtype=np.float32)
            elif args.control_mode == "pd_ee_delta_pos":
                # 3-DOF control: translation only (xyz)
                action = np.array([
                    state.x * args.speed,      # x translation
                    state.y * args.speed,      # y translation
                    state.z * args.speed,      # z translation
                    1.0 if gripper_open else -1.0  # gripper
                ], dtype=np.float32)
            else:
                # Fallback: zero action
                action = np.zeros(action_dim, dtype=np.float32)

            # Handle buttons
            button_left = len(state.buttons) > 0 and state.buttons[0] == 1
            button_right = len(state.buttons) > 1 and state.buttons[1] == 1

            # Left button: toggle gripper (on button press, not hold)
            if button_left and not prev_button_left:
                gripper_open = not gripper_open
                print(f"Gripper: {'OPEN' if gripper_open else 'CLOSED'}")

            # Right button: save trajectory and start new episode
            if button_right and not prev_button_right:
                print(f"✓ Trajectory {num_trajs + 1} completed!")
                num_trajs += 1
                seed += 1
                obs, info = env.reset(seed=seed)
                gripper_open = True
                print()
                print(f"Starting trajectory {num_trajs + 1} (seed={seed})")
                print(f"Task info: {info}")
                print()

            prev_button_left = button_left
            prev_button_right = button_right

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Only auto-reset if episode succeeded (terminated=True)
            # Don't auto-reset on truncation (time limit) - let user control resets
            if terminated:
                # Convert tensor to bool for checking
                term_val = terminated.item() if hasattr(terminated, 'item') else terminated
                success_val = info.get('success', False)
                if hasattr(success_val, 'item'):
                    success_val = success_val.item()

                if term_val or success_val:
                    print(f"✓ Task succeeded! Success={success_val}")
                    print(f"✓ Trajectory {num_trajs + 1} completed!")
                    num_trajs += 1
                    seed += 1
                    obs, info = env.reset(seed=seed)
                    gripper_open = True
                    print()
                    print(f"Starting trajectory {num_trajs + 1} (seed={seed})")
                    print(f"Task info: {info}")
                    print()

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Interrupted by user!")
        print(f"Total trajectories collected: {num_trajs}")
        print("=" * 60)

    finally:
        # Clean up
        print()
        print("Closing SpaceMouse...")
        pyspacemouse.close()
        print("Closing environment...")
        env.close()
        print()
        print(f"✓ All data saved to: {output_dir}")
        print("Done!")


if __name__ == "__main__":
    main()
