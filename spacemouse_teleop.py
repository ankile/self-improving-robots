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
    - Press Ctrl+C in terminal: Quit
"""

import argparse
import gymnasium as gym
import numpy as np
import pyspacemouse
from copy import deepcopy
import mani_skill.envs  # Register ManiSkill environments
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.registration import register_agent


def apply_deadzone(value, threshold):
    """Apply deadzone to SpaceMouse input to filter out noise."""
    return value if abs(value) > threshold else 0.0


def parse_axis_mapping(mapping_str):
    """
    Parse axis mapping string into remapping instructions.

    Examples:
        "xyz" -> no change
        "yxz" -> swap x and y
        "-xyz" -> negate x
        "-x-yz" -> negate x and y

    Returns:
        A function that takes (x, y, z) and returns remapped (x', y', z')
    """
    mapping_str = mapping_str.lower().strip()

    # Build axis lookup
    axis_map = {}
    axis_chars = [c for c in mapping_str if c in 'xyz']

    if len(axis_chars) != 3 or len(set(axis_chars)) != 3:
        raise ValueError(f"Invalid axis mapping: {mapping_str}. Must contain x, y, z exactly once.")

    # Parse each axis and its sign
    for i, char in enumerate(['x', 'y', 'z']):
        # Find where this output axis comes from
        idx = axis_chars.index(char)
        # Check if there's a negative sign before it
        sign_idx = mapping_str.index(char)
        is_negative = sign_idx > 0 and mapping_str[sign_idx - 1] == '-'
        axis_map[i] = (idx, -1.0 if is_negative else 1.0)

    def remap(x, y, z):
        vals = [x, y, z]
        return tuple(vals[axis_map[i][0]] * axis_map[i][1] for i in range(3))

    return remap


def create_custom_panda(stiffness, damping, force_limit):
    """Create a custom Panda robot with tuned control parameters."""
    @register_agent()
    class CustomPanda(Panda):
        uid = "panda_custom_teleop"
        arm_stiffness = stiffness
        arm_damping = damping
        arm_force_limit = force_limit
    return CustomPanda


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
        "render_mode": "rgb_array",
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
    print("Ctrl+C: Quit")
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
    gripper_open = True
    prev_button_left = False

    obs, info = env.reset(seed=0)
    print(f"Environment reset")
    print(f"Task info: {info}")
    print()

    try:
        while True:
            # Open viewer window
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
            if args.control_mode == "pd_ee_delta_pose":
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
            elif args.control_mode == "pd_ee_delta_pos":
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

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Interrupted by user!")
        print("=" * 60)

    finally:
        # Clean up
        print()
        print("Closing SpaceMouse...")
        pyspacemouse.close()
        print("Closing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
