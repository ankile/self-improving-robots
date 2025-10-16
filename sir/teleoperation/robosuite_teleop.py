"""
Minimal teleoperation script for Robosuite environments using SpaceMouse.

This script allows collecting demonstrations from Robosuite environments
for conversion to LeRobot dataset format.

Usage:
    # macOS (REQUIRED - use mjpython for viewer support)
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

    # With camera observations (agentview + wrist camera)
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
        --save-images --cameras "agentview,robot0_eye_in_hand"

    # Linux
    python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

    # Multi-arm
    mjpython -m sir.teleoperation.robosuite_teleop --env TwoArmLift --robot Baxter --config bimanual
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# On macOS, force CGL backend for offscreen rendering when using camera observations
# CGL doesn't require main thread, unlike GLFW
if platform.system() == "Darwin":
    if "MUJOCO_GL" not in os.environ or os.environ.get("MUJOCO_GL") == "":
        os.environ["MUJOCO_GL"] = "cgl"

try:
    import robosuite as suite
    from robosuite import load_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper
except ModuleNotFoundError as exc:
    raise ImportError(
        "Robosuite is required for this script. Install with: pip install robosuite"
    ) from exc

from sir.teleoperation.robosuite_spacemouse import RobosuiteSpaceMouse


def save_example_images(obs, output_dir, camera_keys):
    """
    Save example images from camera observations.

    Args:
        obs (dict): Observation dictionary
        output_dir (Path): Directory to save images
        camera_keys (list): List of camera observation keys
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found. Install with: pip install pillow")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_key in camera_keys:
        img_array = obs[cam_key]
        # Images are typically (H, W, 3) in RGB format
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            # Convert to PIL Image and save
            img = Image.fromarray(img_array.astype(np.uint8))
            img_path = output_dir / f"{cam_key}_example.png"
            img.save(img_path)
            print(f"  Saved {img_path}")
        else:
            print(f"  Unexpected shape for {cam_key}: {img_array.shape}")


def teleop_episode(env, device, max_fr=20, save_images=False, output_dir=None, has_renderer=True):
    """
    Teleoperate a single episode.

    Args:
        env: Robosuite environment
        device: SpaceMouse device
        max_fr (int): Frame rate limit (20 is real-time)
        save_images (bool): Whether to save example images
        output_dir (Path): Directory to save images
        has_renderer (bool): Whether environment has onscreen renderer

    Returns:
        dict: Episode data with observations, actions, rewards, etc.
        None: If reset was requested (to trigger environment recreation)
    """
    obs = env.reset()
    if has_renderer:
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
    print("Right button to reset, Ctrl+C to quit.")
    print(f"Observation keys: {list(obs.keys())}")

    # Check for camera observations and save example if requested
    if save_images and output_dir is not None:
        camera_keys = [k for k in obs.keys() if k.endswith("_image")]
        if camera_keys:
            print(f"Camera observations found: {camera_keys}")
            save_example_images(obs, output_dir, camera_keys)
        else:
            print("No camera observations found in obs dict")
    print()

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
        if has_renderer:
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

    # Parse camera names if provided
    camera_names = None
    if args.cameras:
        camera_names = [cam.strip() for cam in args.cameras.split(",")]

    # Enable camera observations if requested
    use_camera_obs = args.save_images and camera_names is not None
    has_offscreen_renderer = use_camera_obs  # Must be True for camera obs
    has_renderer = not args.headless  # Only show viewer if not headless

    # Note: On macOS, CGL backend allows both viewer + camera obs simultaneously

    # Create environment config
    env_config = {
        "env_name": args.env,
        "robots": args.robot,
        "controller_configs": controller_config,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": has_offscreen_renderer,  # Explicitly set
        "render_camera": args.camera,
        "ignore_done": True,
        "use_camera_obs": use_camera_obs,
        "reward_shaping": True,
        "control_freq": 20,
        "hard_reset": False,
    }

    # Add camera configuration if enabled
    if use_camera_obs:
        env_config["camera_names"] = camera_names
        env_config["camera_heights"] = args.camera_height
        env_config["camera_widths"] = args.camera_width
        env_config["camera_depths"] = False  # RGB only
        print(f"Camera observations enabled: {camera_names}")
        print(f"Resolution: {args.camera_width}x{args.camera_height}")

    # Handle TwoArm environments
    if "TwoArm" in args.env:
        env_config["env_configuration"] = args.config

    # Create environment
    env = suite.make(**env_config)

    # Wrap with visualization
    env = VisualizationWrapper(env, indicator_configs=None)

    return env


def main():
    # Parse args first to check if headless mode
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
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save example images from camera observations",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera names for observations (e.g., 'agentview,robot0_eye_in_hand')",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=256,
        help="Camera image height",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=256,
        help="Camera image width",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./robosuite_images",
        help="Directory to save images",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without onscreen viewer (allows camera obs on macOS)",
    )
    args = parser.parse_args()

    # Check if running on macOS with regular python (not mjpython)
    if platform.system() == "Darwin" and not args.headless:
        # mjpython sets _MJPYTHON attribute in mujoco.viewer module
        try:
            import mujoco.viewer
            is_mjpython = hasattr(mujoco.viewer, '_MJPYTHON')
        except (ImportError, AttributeError):
            is_mjpython = False

        if not is_mjpython:
            print("=" * 70)
            print("ERROR: On macOS, you must use 'mjpython' instead of 'python'")
            print("=" * 70)
            print("\nThe MuJoCo viewer requires GUI operations on the main thread,")
            print("which is only supported by MuJoCo's mjpython wrapper on macOS.\n")
            print("Please run:")
            # Reconstruct the command properly
            module_args = [arg for arg in sys.argv[1:]]
            if module_args:
                print(f"  mjpython -m sir.teleoperation.robosuite_teleop {' '.join(module_args)}")
            else:
                print("  mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda")
            print("\nAlternatively, use --headless mode with regular python:")
            print("  python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda --headless")
            print("\nIf mjpython is not found, ensure mujoco is installed:")
            print("  pip install mujoco")
            print("=" * 70)
            sys.exit(1)

    # Info message if using camera observations with viewer on macOS
    if platform.system() == "Darwin" and args.save_images and not args.headless:
        print("=" * 70)
        print("INFO: macOS with camera observations + onscreen viewer")
        print("=" * 70)
        print("\nUsing CGL backend for offscreen rendering (not GLFW).")
        print("This allows simultaneous onscreen viewer and camera observations.")
        print("CGL contexts are not tied to the main thread on macOS.")
        print("=" * 70)

    # Initialize SpaceMouse once (persists across episodes)
    print("\nInitializing SpaceMouse...")
    device = RobosuiteSpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    # Setup output directory for images
    output_dir = Path(args.output_dir) if args.save_images else None

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

            episode_data = teleop_episode(
                env, device, max_fr=args.max_fr,
                save_images=args.save_images,
                output_dir=output_dir,
                has_renderer=not args.headless
            )

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
