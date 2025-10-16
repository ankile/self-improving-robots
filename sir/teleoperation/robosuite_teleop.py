"""
Minimal teleoperation script for Robosuite environments using SpaceMouse.

This script allows collecting demonstrations from Robosuite environments
for conversion to LeRobot dataset format.

Usage:
    python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda
    python -m sir.teleoperation.robosuite_teleop --env TwoArmLift --robot Baxter --config bimanual
"""

import argparse
import time

import numpy as np

try:
    import robosuite as suite
    from robosuite import load_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper
except ModuleNotFoundError as exc:
    raise ImportError(
        "Robosuite is required for this script. Install with: pip install robosuite"
    ) from exc

from sir.teleoperation.robosuite_spacemouse import RobosuiteSpaceMouse


def teleop_episode(env, device, max_fr=20):
    """
    Teleoperate a single episode.

    Args:
        env: Robosuite environment
        device: SpaceMouse device
        max_fr (int): Frame rate limit (20 is real-time)

    Returns:
        dict: Episode data with observations, actions, rewards, etc.
        None: If reset was requested (to trigger environment recreation)
    """
    obs = env.reset()
    env.render()

    device.start_control()

    # Storage for episode data
    episode_data = {
        "observations": [obs],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    print("\nEpisode started. Use SpaceMouse to control the robot.")
    print("Right button to reset, Ctrl+C to quit.\n")

    step_count = 0

    while True:
        start = time.time()

        # Check for reset - need to recreate environment due to viewer limitation
        if device.reset_requested:
            print(f"\nReset requested. Episode length: {step_count} steps")
            return None  # Signal that env needs to be recreated

        # Get SpaceMouse control
        control = device.control
        gripper = device.control_gripper

        # Scale controls (base 0.005 scaling factor)
        dpos = control[:3] * 0.005 * device.pos_sensitivity
        raw_drot = control[3:6] * 0.005 * device.rot_sensitivity  # [roll, pitch, yaw]

        # Apply coordinate frame transformation (from Device.input2action)
        # Robot expects [pitch, roll, -yaw] not [roll, pitch, yaw]
        drot = raw_drot[[1, 0, 2]]  # Reorder to [pitch, roll, yaw]
        drot[2] = -drot[2]           # Flip yaw sign → [pitch, roll, -yaw]

        # Apply postprocessing: multiply by scale factors and clip
        # (matches Device._postprocess_device_outputs)
        dpos = np.clip(dpos * 125, -1, 1)
        drot = np.clip(drot * 50, -1, 1)

        # Gripper action: map 0/1 to -1/1 (open/close)
        gripper_action = 1.0 if gripper == 1 else -1.0

        # Construct action
        # For single-arm robots: [dx, dy, dz, dpitch, droll, -dyaw, gripper]
        action = np.concatenate([dpos, drot, [gripper_action]])

        # Step environment
        obs, reward, done, info = env.step(action)
        env.render()

        # Store data
        episode_data["observations"].append(obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)

        step_count += 1

        # Check for task completion
        if done:
            print(f"\nEpisode completed! Length: {step_count} steps")
            break

        # Limit frame rate
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    return episode_data


def create_env(args):
    """
    Create and configure the Robosuite environment.

    Args:
        args: Parsed command line arguments

    Returns:
        Wrapped Robosuite environment
    """
    # Setup controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robot,
    )

    # Create environment config
    env_config = {
        "env_name": args.env,
        "robots": args.robot,
        "controller_configs": controller_config,
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "render_camera": args.camera,
        "ignore_done": True,
        "use_camera_obs": False,
        "reward_shaping": True,
        "control_freq": 20,
        "hard_reset": False,
    }

    # Handle TwoArm environments
    if "TwoArm" in args.env:
        env_config["env_configuration"] = args.config

    # Create environment
    env = suite.make(**env_config)

    # Wrap with visualization
    env = VisualizationWrapper(env, indicator_configs=None)

    return env


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperate Robosuite environments with SpaceMouse"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Lift",
        help="Environment name (e.g., Lift, PickPlaceCan, TwoArmLift)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="Panda",
        help="Robot name (e.g., Panda, Sawyer, Baxter, UR5e)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Environment configuration (for TwoArm envs: bimanual, parallel, opposed)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller type (default uses robot's default controller)",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="Position control sensitivity",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="Rotation control sensitivity",
    )
    parser.add_argument(
        "--max-fr",
        type=int,
        default=20,
        help="Frame rate limit (20 is real-time)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name for rendering",
    )
    args = parser.parse_args()

    # Initialize SpaceMouse once (persists across episodes)
    print("\nInitializing SpaceMouse...")
    device = RobosuiteSpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    print("\nSetup complete! Ready to teleoperate.")
    print("=" * 60)

    try:
        episode_count = 0
        env = None

        while True:
            episode_count += 1
            print(f"\n{'='*60}")
            print(f"Episode {episode_count}")
            print(f"{'='*60}")

            # Create fresh environment for each episode (required for viewer)
            if env is not None:
                env.close()

            print(f"Creating environment: {args.env} with robot: {args.robot}")
            env = create_env(args)

            episode_data = teleop_episode(env, device, max_fr=args.max_fr)

            # If None returned, user requested reset - continue to next episode
            if episode_data is None:
                print("\nStarting new episode...")
                time.sleep(0.5)
                continue

            # Print episode summary
            total_reward = sum(episode_data["rewards"])
            print(f"\nEpisode Summary:")
            print(f"  Steps: {len(episode_data['actions'])}")
            print(f"  Total Reward: {total_reward:.3f}")
            print(f"  Success: {episode_data['dones'][-1] if episode_data['dones'] else False}")

            # TODO: Save episode data to LeRobot format
            # This is where you would convert and save to LeRobot dataset

            print("\nPress Ctrl+C to quit, or continue for another episode...")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        device.close()
        if env is not None:
            env.close()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
