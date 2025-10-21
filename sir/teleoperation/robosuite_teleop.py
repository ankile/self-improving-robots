"""
Minimal teleoperation script for Robosuite environments using SpaceMouse.

This script allows collecting demonstrations from Robosuite environments
and saving them to LeRobot dataset format.

Usage:
    # macOS (REQUIRED - use mjpython for viewer support)
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

    # With camera observations (agentview + wrist camera)
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
        --cameras "agentview,robot0_eye_in_hand"

    # Save demonstrations to LeRobot dataset
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
        --save-data --dataset-name my_demos \
        --cameras "agentview,robot0_eye_in_hand"

    # Save with auto-generated dataset name (env_robot_timestamp)
    mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
        --save-data --cameras "agentview"

    # Linux
    python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

    # Multi-arm
    mjpython -m sir.teleoperation.robosuite_teleop --env TwoArmLift --robot Baxter --config bimanual

Controls:
    - SpaceMouse: Control robot end-effector (6-DOF)
    - Left button: Toggle gripper
    - '1' key: Mark episode as SUCCESS and save to dataset (if --save-data enabled)
    - '0' key: Mark episode as FAILURE (reset without saving)
    - Ctrl+C: Quit

Data Collection:
    - Use --save-data flag to enable saving to LeRobot dataset
    - Press '1' to save successful episodes, '0' to discard failures
    - Dataset includes robot state, actions, rewards, and camera images
    - Compatible with HuggingFace LeRobot for training
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# On macOS, force CGL backend for offscreen rendering when using camera observations
# CGL doesn't require main thread, unlike GLFW
if platform.system() == "Darwin":
    if "MUJOCO_GL" not in os.environ or os.environ.get("MUJOCO_GL") == "":
        os.environ["MUJOCO_GL"] = "cgl"

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import robosuite.macros as macros

from sir.teleoperation.robosuite_spacemouse import RobosuiteSpaceMouse
from sir.teleoperation.utils import KeyboardListener

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# IMPORTANT: Set image convention to OpenCV (origin at top-left) instead of OpenGL (origin at bottom-left)
# OpenGL (default): origin at bottom-left, Y-up → images rendered upside down
# OpenCV: origin at top-left, Y-down → standard image format (PIL, cv2, LeRobot)
# This must be set BEFORE creating the environment to take effect
# Without this, all camera observations would be vertically flipped!
macros.IMAGE_CONVENTION = "opencv"


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


def teleop_episode(
    env,
    device,
    kbd_listener,
    max_fr=20,
    save_data=False,
    save_example_images_flag=False,
    output_dir=None,
    has_renderer=True,
    camera_names=None,
):
    """
    Teleoperate a single episode.

    Args:
        env: Robosuite environment
        device: SpaceMouse device
        kbd_listener: KeyboardListener for detecting key presses
        max_fr (int): Frame rate limit (20 is real-time)
        save_data (bool): Whether to save data to LeRobot dataset
        save_example_images_flag (bool): Whether to save example images
        output_dir (Path): Directory to save example images
        has_renderer (bool): Whether environment has onscreen renderer
        camera_names (list): List of camera names for observations

    Returns:
        tuple: (episode_data, episode_success) where:
            - episode_data is dict with observations, actions, rewards, etc. (None if reset without completion)
            - episode_success is True/False/None (None if reset without marking success/failure)
    """
    obs = env.reset()
    if has_renderer:
        env.render()

    device.start_control()

    # Storage for episode data
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    # Extract camera images if present
    if camera_names:
        camera_keys = [f"{cam}_image" for cam in camera_names]
        for cam_key in camera_keys:
            if cam_key in obs:
                episode_data[cam_key] = []
    else:
        camera_keys = []

    # Extract state observation
    # Use minimal task-relevant state: end effector pose + gripper
    # No joints (IK handled by controller), no object state, no velocities
    state_keys = [
        "robot0_eef_pos",        # 3 dims - end effector position
        "robot0_eef_quat",       # 4 dims - end effector orientation (quaternion)
        "robot0_gripper_qpos",   # 2 dims - gripper finger positions
        # Total: 9 dims
    ]
    state_keys = [k for k in state_keys if k in obs]
    state_obs = np.concatenate([obs[k].flatten() for k in state_keys])

    episode_data["observations"].append(state_obs)

    # Store initial camera observations if saving data
    if save_data:
        for cam_key in camera_keys:
            if cam_key in obs:
                episode_data[cam_key].append(obs[cam_key].copy())

    print("\nEpisode started. Use SpaceMouse to control the robot.")
    print("Press '1' for SUCCESS and save, '0' for FAILURE (no save), Ctrl+C to quit.")
    print(f"Observation keys: {list(obs.keys())}")

    # Check for camera observations and save example if requested
    if save_example_images_flag and output_dir is not None:
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

        # Check for keyboard input
        key = kbd_listener.read_key()
        if key == "1":
            print()
            print("=" * 60)
            print(f"SUCCESS! Episode length: {step_count} steps")
            print("=" * 60)
            print()
            return (episode_data, True)
        elif key == "0":
            print()
            print("=" * 60)
            print(f"FAILURE! Episode length: {step_count} steps")
            print("=" * 60)
            print()
            return (episode_data, False)

        # Check for reset - need to recreate environment due to viewer limitation
        if device.reset_requested:
            print(f"\nReset requested. Episode length: {step_count} steps")
            return (None, None)  # Signal that env needs to be recreated

        # Get SpaceMouse control
        control = device.control
        gripper = device.control_gripper

        # Scale controls (base 0.005 scaling factor)
        dpos = control[:3] * 0.005 * device.pos_sensitivity
        raw_drot = control[3:6] * 0.005 * device.rot_sensitivity  # [roll, pitch, yaw]

        # Apply coordinate frame transformation (from Device.input2action)
        # Robot expects [pitch, roll, -yaw] not [roll, pitch, yaw]
        drot = raw_drot[[1, 0, 2]]  # Reorder to [pitch, roll, yaw]
        drot[2] = -drot[2]  # Flip yaw sign → [pitch, roll, -yaw]

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
        # Extract state observation (using same logic as initial obs)
        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        state_keys = [k for k in state_keys if k in obs]
        state_obs = np.concatenate([obs[k].flatten() for k in state_keys])
        episode_data["observations"].append(state_obs)
        episode_data["actions"].append(action.copy())
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)

        # Store camera observations if saving data
        if save_data:
            for cam_key in camera_keys:
                if cam_key in obs:
                    episode_data[cam_key].append(obs[cam_key].copy())

        step_count += 1

        # Check for task completion
        if done:
            print(f"\nEpisode completed! Length: {step_count} steps")
            print("Press '1' to mark as SUCCESS and save, '0' to mark as FAILURE")
            # Don't break - wait for user to mark success/failure
            # break

        # Limit frame rate
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # This should not be reached (user should press 1 or 0 to exit)
    return (episode_data, None)


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
    # Enable cameras if either saving images OR saving data (with cameras specified)
    use_camera_obs = (args.save_images or args.save_data) and camera_names is not None
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
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save teleoperation data to LeRobotDataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to save dataset (default: ./data)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset. If not specified, will be generated as {env}_{robot}_{timestamp}",
    )
    args = parser.parse_args()

    # Check if LeRobotDataset is available when saving data
    if args.save_data and LeRobotDataset is None:
        print("=" * 70)
        print("ERROR: LeRobotDataset not available")
        print("=" * 70)
        print("\nTo save data, you need to install lerobot:")
        print("  pip install lerobot")
        print("=" * 70)
        sys.exit(1)

    # Check if running on macOS with regular python (not mjpython)
    if platform.system() == "Darwin" and not args.headless:
        # mjpython sets _MJPYTHON attribute in mujoco.viewer module
        try:
            import mujoco.viewer

            is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON")
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
            print(
                "  python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda --headless"
            )
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

    # Initialize keyboard listener
    print("\nInitializing keyboard listener...")
    kbd_listener = KeyboardListener()
    print("✓ Keyboard listener initialized (works regardless of window focus)")

    # Setup output directory for example images
    output_dir = Path(args.output_dir) if args.save_images else None

    # Parse camera names if provided
    camera_names = None
    if args.cameras:
        camera_names = [cam.strip() for cam in args.cameras.split(",")]

    # Data saving setup
    dataset = None
    saved_episode_count = 0
    if args.save_data:
        dataset_name = args.dataset_name
        if dataset_name is None:
            # Generate dataset name from env, robot and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = args.env.lower().replace(" ", "_")
            robot_name = args.robot.lower()
            dataset_name = f"{env_name}_{robot_name}_{timestamp}"

        dataset_path = Path(args.dataset_path) / dataset_name
        print(f"\nData saving: ENABLED")
        print(f"Dataset path: {dataset_path}")
        print(f"Dataset name: {dataset_name}")
    else:
        print(f"\nData saving: DISABLED")

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

            episode_data, episode_success = teleop_episode(
                env,
                device,
                kbd_listener,
                max_fr=args.max_fr,
                save_data=args.save_data,
                save_example_images_flag=args.save_images,
                output_dir=output_dir,
                has_renderer=not args.headless,
                camera_names=camera_names,
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

            # Save episode data to LeRobot dataset if marked as success
            if args.save_data and episode_success and len(episode_data["actions"]) > 0:
                # Initialize dataset on first save
                if dataset is None:
                    # Get observation shape from first observation
                    obs_shape = episode_data["observations"][0].shape[0]
                    action_shape = episode_data["actions"][0].shape[0]

                    # Build features dict
                    features = {
                        "observation.state": {
                            "dtype": "float32",
                            "shape": (obs_shape,),
                            "names": [f"state_{i}" for i in range(obs_shape)],
                        },
                        "action": {
                            "dtype": "float32",
                            "shape": (action_shape,),
                            "names": [f"action_{i}" for i in range(action_shape)],
                        },
                    }

                    # Add camera features if present
                    for cam_name in camera_names or []:
                        cam_key = f"{cam_name}_image"
                        if cam_key in episode_data:
                            img_shape = episode_data[cam_key][0].shape
                            features[f"observation.images.{cam_name}"] = {
                                "dtype": "image",  # Use "image" not "uint8"
                                "shape": img_shape,
                                "names": ["height", "width", "channels"],
                            }

                    dataset = LeRobotDataset.create(
                        repo_id=dataset_name,
                        fps=20,  # Control frequency
                        root=str(dataset_path),  # Full path including dataset name
                        robot_type=args.robot.lower(),
                        features=features,
                    )
                    print(f"✓ Dataset initialized at {dataset_path}")

                # Add frames using add_frame() method (proper LeRobot API)
                task_name = f"{args.env}_{args.robot}"
                num_frames = len(episode_data["actions"])

                for i in range(num_frames):
                    frame = {
                        "task": task_name,
                        "observation.state": episode_data["observations"][i].astype(np.float32),
                        "action": episode_data["actions"][i].astype(np.float32),
                    }

                    # Add camera observations if present
                    for cam_name in camera_names or []:
                        cam_key = f"{cam_name}_image"
                        if cam_key in episode_data:
                            frame[f"observation.images.{cam_name}"] = episode_data[cam_key][i]

                    dataset.add_frame(frame)

                # Save the episode (no arguments - uses internal episode_buffer)
                dataset.save_episode()
                saved_episode_count += 1
                print(f"✓ Episode saved to dataset (Total saved: {saved_episode_count})")
            elif args.save_data and not episode_success:
                print("Episode marked as failure - not saved to dataset")

            print("\nReady for next episode...")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("\nCleaning up...")
        if args.save_data and dataset is not None:
            print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
            dataset.finalize()
        print("Stopping keyboard listener...")
        kbd_listener.close()
        print("Closing SpaceMouse...")
        device.close()
        if env is not None:
            print("Closing environment...")
            env.close()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
