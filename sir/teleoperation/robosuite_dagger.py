"""
DAgger (Dataset Aggregation) data collection for Robosuite with LeRobot policies.

This script enables human-in-the-loop data collection for iterative policy improvement:
1. Loads a pre-trained LeRobot policy from W&B artifact
2. Runs the policy autonomously (policy rollout)
3. Allows human to intervene when policy fails (press 'h')
4. Human corrects using SpaceMouse teleoperation
5. Saves both policy and human trajectories with appropriate labels

Dataset Structure:
- Original demo dataset: Kept intact (e.g., "square-v1")
- DAgger dataset: New data with labels (e.g., "square-dagger-v1")
  - source: 0=policy, 1=human (encoded as int64 array)
  - success: 0=False, 1=True (encoded as int64 array)

Label Meanings:
  - source=0, success=0: Policy failed (user intervened with 'h')
  - source=0, success=1: Policy succeeded (completed task without intervention)
  - source=1, success=1: Human correction succeeded (user pressed '1')
  - source=1, success=0: Not used (human corrections are only saved if marked success)

Usage:
    # Basic DAgger collection for NutAssemblySquare
    mjpython -m sir.teleoperation.robosuite_dagger \
        --wandb-artifact "self-improving/act-training/act-ankile_square-v1-best-step-9000:v0" \
        --env NutAssemblySquare \
        --robot Panda \
        --cameras "agentview,robot0_eye_in_hand" \
        --dataset-name square-dagger-v1 \
        --wandb-project square-dagger

    # With custom parameters
    mjpython -m sir.teleoperation.robosuite_dagger \
        --wandb-artifact "my-project/act-lift-best:v3" \
        --env Lift \
        --robot Panda \
        --cameras "agentview" \
        --dataset-path ./data \
        --max-steps 400 \
        --wandb-project my-dagger-project

    # Auto-save on task success (no manual intervention needed for successful episodes)
    mjpython -m sir.teleoperation.robosuite_dagger \
        --wandb-artifact "my-project/act-lift-best:v3" \
        --env Lift \
        --robot Panda \
        --cameras "agentview" \
        --dataset-name lift-dagger-v1 \
        --auto-save-on-success

Controls:
    During policy rollout:
    - 'h' key: Trigger human intervention (policy failed, switch to spacemouse)
    - With --auto-save-on-success: Successful episodes save automatically (no intervention needed)

    During spacemouse mode:
    - SpaceMouse: Control robot end-effector (6-DOF)
    - Left button: Toggle gripper
    - '1' key: Mark correction as SUCCESS and save
    - '0' key: Discard correction (no save)

    Between episodes:
    - 'n' key: Start next episode
    - Ctrl+C: Quit
"""

import argparse
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Suppress Pydantic warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)

import numpy as np
import torch
import wandb

# Set MuJoCo GL backend to CGL on macOS
if platform.system() == "Darwin":
    if "MUJOCO_GL" not in os.environ or os.environ.get("MUJOCO_GL") == "":
        os.environ["MUJOCO_GL"] = "cgl"

# Robosuite imports
import robosuite as suite
import robosuite.macros as macros
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

# SIR imports
from sir.teleoperation.robosuite_spacemouse import RobosuiteSpaceMouse
from sir.teleoperation.utils import KeyboardListener
from sir.training.load_pretrained import load_policy_from_wandb

# Set image convention to OpenCV
macros.IMAGE_CONVENTION = "opencv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DAgger data collection with policy rollouts and human interventions"
    )

    # Policy args
    parser.add_argument(
        "--wandb-artifact",
        type=str,
        required=True,
        help="W&B artifact identifier (e.g., 'act-square-v1-best:latest' or 'project/name:version')",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional, uses default if not specified)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="square-dagger",
        help="W&B project name (default: square-dagger)",
    )

    # Environment args
    parser.add_argument(
        "--env",
        type=str,
        default="Stack",
        help="Robosuite environment name (default: Stack)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="Panda",
        help="Robot name (default: Panda)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller type (default uses robot's default controller)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Environment configuration (for TwoArm envs: bimanual, parallel, opposed)",
    )

    # Camera args
    parser.add_argument(
        "--cameras",
        type=str,
        required=True,
        help="Comma-separated camera names (e.g., 'agentview,robot0_eye_in_hand'). Must match training.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Camera image height (auto-detected from policy if not specified)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="Camera image width (auto-detected from policy if not specified)",
    )

    # Policy rollout args
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum steps per episode (default: 400)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        help="Device for policy inference (cuda/mps/cpu)",
    )

    # SpaceMouse args
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="Position control sensitivity for spacemouse",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="Rotation control sensitivity for spacemouse",
    )

    # Dataset args
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to save dataset (default: ./data)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the DAgger dataset (e.g., 'square-dagger-v1')",
    )

    # System args
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without onscreen viewer",
    )
    parser.add_argument(
        "--visual-aids",
        action="store_true",
        help="Enable visual aids (indicators) in the environment",
    )
    parser.add_argument(
        "--max-fr",
        type=int,
        default=20,
        help="Frame rate limit (default: 20 Hz, real-time)",
    )
    parser.add_argument(
        "--auto-save-on-success",
        action="store_true",
        help="Automatically save policy episode when task completion is detected (no need to intervene)",
    )

    return parser.parse_args()


def create_env(
    env_name: str,
    robot_name: str,
    camera_names: list,
    camera_height: int,
    camera_width: int,
    controller: Optional[str] = None,
    config: str = "default",
    has_renderer: bool = True,
    visual_aids: bool = False,
):
    """Create Robosuite environment for DAgger data collection."""
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot_name,
    )

    use_camera_obs = camera_names is not None and len(camera_names) > 0

    env_config = {
        "env_name": env_name,
        "robots": robot_name,
        "controller_configs": controller_config,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": use_camera_obs,
        "render_camera": camera_names[0] if camera_names else "agentview",
        "ignore_done": True,
        "use_camera_obs": use_camera_obs,
        "reward_shaping": True,
        "control_freq": 20,
        "hard_reset": False,
    }

    if use_camera_obs:
        env_config["camera_names"] = camera_names
        env_config["camera_heights"] = camera_height
        env_config["camera_widths"] = camera_width
        env_config["camera_depths"] = False

    if "TwoArm" in env_name:
        env_config["env_configuration"] = config

    env = suite.make(**env_config)

    if visual_aids:
        env = VisualizationWrapper(env, indicator_configs=None)

    return env


def extract_camera_config_from_policy(
    policy: ACTPolicy,
) -> Tuple[list, Optional[int], Optional[int]]:
    """
    Extract camera configuration from policy config.

    Returns:
        Tuple of (camera_names, height, width)
    """
    camera_names = []
    camera_height = None
    camera_width = None

    if hasattr(policy.config, "image_features") and policy.config.image_features:
        for img_key in policy.config.image_features:
            if img_key.startswith("observation.images."):
                cam_name = img_key.replace("observation.images.", "")
                camera_names.append(cam_name)

        # Get image shape from config
        if hasattr(policy.config, "input_features"):
            for key, feature in policy.config.input_features.items():
                if key.startswith("observation.images.") and hasattr(feature, "shape"):
                    # Shape is (C, H, W)
                    if len(feature.shape) == 3:
                        c, h, w = feature.shape
                        camera_height = h
                        camera_width = w
                        break

    return camera_names, camera_height, camera_width


def policy_rollout_episode(
    env,
    policy: ACTPolicy,
    preprocessor: Any,
    postprocessor: Any,
    action_stats: Dict[str, Any],
    kbd_listener: KeyboardListener,
    camera_names: list,
    device: str,
    max_steps: int = 400,
    max_fr: int = 20,
    has_renderer: bool = True,
    auto_save_on_success: bool = False,
) -> Tuple[Optional[Dict], bool, bool]:
    """
    Run policy rollout with option for human intervention.

    Args:
        auto_save_on_success: Automatically end episode and save when task success is detected

    Returns:
        Tuple of (episode_data, intervention_triggered, task_success)
        - episode_data: Dict with observations, actions, etc. (None if no data)
        - intervention_triggered: True if user pressed 'h' to intervene
        - task_success: True if task completed successfully before intervention
    """
    obs = env.reset()
    if has_renderer:
        env.render()

    policy.reset()

    # Storage for episode data
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    # Camera image keys
    camera_keys = [f"{cam}_image" for cam in camera_names] if camera_names else []
    for cam_key in camera_keys:
        if cam_key in obs:
            episode_data[cam_key] = []

    # Prepare action denormalization stats
    if isinstance(action_stats["mean"], torch.Tensor):
        action_mean = action_stats["mean"].to(device=device, dtype=torch.float32)
        action_std = action_stats["std"].to(device=device, dtype=torch.float32)
    else:
        action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.float32)
        action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.float32)

    print("\n" + "=" * 60)
    print("POLICY ROLLOUT MODE")
    print("=" * 60)
    print("Policy is controlling the robot...")
    if auto_save_on_success:
        print("Auto-save ENABLED: Episode will save automatically when task success is detected")
        print("Press 'h' for manual INTERVENTION (if needed)")
    else:
        print("Press 'h' to trigger human INTERVENTION")
    print("=" * 60)
    print()

    step_count = 0
    intervention_triggered = False
    task_success = False
    success_announced = False  # Track if we've announced task success

    for step_count in range(max_steps):
        start = time.time()

        # Check for intervention signal
        key = kbd_listener.read_key()
        if key == "h":
            print("\n" + "!" * 60)
            print("HUMAN INTERVENTION TRIGGERED")
            print("!" * 60)
            intervention_triggered = True
            break

        # Extract state observation
        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        state_keys = [k for k in state_keys if k in obs]
        state = np.concatenate([obs[k].flatten() for k in state_keys])

        # Prepare observation for policy
        obs_dict = {}

        # Add state
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        obs_dict["observation.state"] = state_tensor

        # Add camera observations
        for cam_name in camera_names:
            img_key = f"{cam_name}_image"
            if img_key in obs:
                img = obs[img_key]
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                img_tensor = img_tensor.unsqueeze(0).to(device)
                obs_dict[f"observation.images.{cam_name}"] = img_tensor

        # Apply preprocessing
        obs_dict = preprocessor(obs_dict)

        # Get action from policy
        with torch.no_grad():
            action_tensor = policy.select_action(obs_dict)
            # Denormalize
            action_denorm = action_tensor * action_std + action_mean
            action = action_denorm.cpu().numpy()[0]

        # Step environment
        obs, reward, done, info = env.step(action)
        if has_renderer:
            env.render()

        # Check for task success and announce once (BEFORE storing data)
        if not success_announced and hasattr(env, '_check_success'):
            task_success = env._check_success()
            if task_success:
                success_announced = True
                print()
                print("*" * 60)
                print("TASK SUCCESSFUL! Environment detected task completion!")
                print("*" * 60)
                print()

                # Automatically save and end episode if enabled
                if auto_save_on_success:
                    print("Auto-save enabled - saving episode as SUCCESS")
                    print("=" * 60)
                    print(f"POLICY SUCCEEDED! Task completed in {step_count + 1} steps")
                    print("=" * 60)
                    print()
                    # Store this final step before breaking
                    episode_data["observations"].append(state.copy())
                    episode_data["actions"].append(action.copy())
                    episode_data["rewards"].append(reward)
                    episode_data["dones"].append(done)
                    for cam_key in camera_keys:
                        if cam_key in obs:
                            episode_data[cam_key].append(obs[cam_key].copy())
                    break

        # Store data
        episode_data["observations"].append(state.copy())
        episode_data["actions"].append(action.copy())
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done)

        # Store camera observations
        for cam_key in camera_keys:
            if cam_key in obs:
                episode_data[cam_key].append(obs[cam_key].copy())

        # Limit frame rate
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    print(f"\nPolicy rollout ended: {step_count + 1} steps collected")

    if len(episode_data["actions"]) == 0:
        return None, intervention_triggered, task_success

    return episode_data, intervention_triggered, task_success


def spacemouse_correction_episode(
    env,
    device: RobosuiteSpaceMouse,
    kbd_listener: KeyboardListener,
    camera_names: list,
    max_fr: int = 20,
    has_renderer: bool = True,
    record_gripper_motion: bool = True,
    gripper_vel_threshold: float = 0.01,
    auto_save_on_success: bool = False,
) -> Tuple[Optional[Dict], bool]:
    """
    Collect human correction using SpaceMouse.

    Args:
        auto_save_on_success: Automatically save when task completion is detected

    Returns:
        Tuple of (episode_data, success)
        - episode_data: Dict with trajectory data (None if discarded)
        - success: True if user marked as success (pressed '1' or auto-saved)
    """
    # Get current observation (don't reset - continue from where policy left off)
    obs = env._get_observations()
    if has_renderer:
        env.render()

    device.start_control()
    device.reset_gripper()

    # Storage for correction data
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    camera_keys = [f"{cam}_image" for cam in camera_names] if camera_names else []
    for cam_key in camera_keys:
        if cam_key in obs:
            episode_data[cam_key] = []

    print("\n" + "=" * 60)
    print("SPACEMOUSE CORRECTION MODE")
    print("=" * 60)
    print("Use SpaceMouse to correct the robot's behavior")
    if auto_save_on_success:
        print("Auto-save ENABLED: Episode will save automatically when task success is detected")
        print("Press '0' to DISCARD and skip")
    else:
        print("Press '1' to mark as SUCCESS and save")
        print("Press '0' to DISCARD and skip")
    print("=" * 60)
    print()

    step_count = 0
    prev_gripper = device.control_gripper
    gripper_is_moving = False
    target_gripper_state = prev_gripper
    success_announced = False  # Track if we've announced task success

    while True:
        start = time.time()

        # Check for keyboard input
        key = kbd_listener.read_key()
        if key == "1":
            print("\n" + "=" * 60)
            print(f"SUCCESS! Correction saved ({step_count} steps)")
            print("=" * 60)
            return episode_data, True
        elif key == "0":
            print("\n" + "=" * 60)
            print(f"DISCARDED! Correction not saved ({step_count} steps)")
            print("=" * 60)
            return None, False

        # Get SpaceMouse control
        control = device.control
        gripper = device.control_gripper

        # Scale controls (matching robosuite_teleop.py)
        dpos = control[:3] * 0.005 * device.pos_sensitivity
        raw_drot = control[3:6] * 0.005 * device.rot_sensitivity

        # Coordinate frame transformation
        drot = raw_drot[[1, 0, 2]]
        drot[2] = -drot[2]

        # Postprocessing
        dpos = np.clip(dpos * 125, -1, 1)
        drot = np.clip(drot * 50, -1, 1)

        # Detect gripper change
        if gripper != prev_gripper:
            if record_gripper_motion:
                gripper_is_moving = True
                target_gripper_state = gripper

        # Gripper action
        gripper_action = 1.0 if gripper == 1 else -1.0

        # Construct action
        action = np.concatenate([dpos, drot, [gripper_action]])

        # Step environment
        obs, reward, done, info = env.step(action)
        if has_renderer:
            env.render()

        # Check for task success and announce once
        if not success_announced and hasattr(env, '_check_success'):
            task_success = env._check_success()
            if task_success:
                success_announced = True
                print()
                print("*" * 60)
                print("TASK SUCCESSFUL! Environment detected task completion!")
                print("*" * 60)
                print()

                # Automatically save and end episode if enabled
                if auto_save_on_success:
                    print("Auto-save enabled - saving episode as SUCCESS")
                    print("=" * 60)
                    print(f"SUCCESS! Correction saved ({step_count} steps)")
                    print("=" * 60)
                    print()
                    # Store this final step if there's input
                    has_input = np.any(np.abs(control) > 0.01) or (gripper != prev_gripper) or gripper_is_moving
                    if has_input:
                        state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
                        state_keys = [k for k in state_keys if k in obs]
                        state_obs = np.concatenate([obs[k].flatten() for k in state_keys])
                        episode_data["observations"].append(state_obs)
                        episode_data["actions"].append(action.copy())
                        episode_data["rewards"].append(reward)
                        episode_data["dones"].append(done)
                        for cam_key in camera_keys:
                            if cam_key in obs:
                                episode_data[cam_key].append(obs[cam_key].copy())
                    return episode_data, True

        # Check gripper motion
        if gripper_is_moving and record_gripper_motion:
            gripper_qvel_key = "robot0_gripper_qvel"
            if gripper_qvel_key in obs:
                gripper_qvel = obs[gripper_qvel_key]
                max_qvel = np.max(np.abs(gripper_qvel))
                if max_qvel < gripper_vel_threshold:
                    gripper_is_moving = False

        # Check if there's actual input
        has_input = np.any(np.abs(control) > 0.01) or (gripper != prev_gripper) or gripper_is_moving

        # Store data when there's input
        if has_input:
            state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            state_keys = [k for k in state_keys if k in obs]
            state_obs = np.concatenate([obs[k].flatten() for k in state_keys])

            episode_data["observations"].append(state_obs)
            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)

            for cam_key in camera_keys:
                if cam_key in obs:
                    episode_data[cam_key].append(obs[cam_key].copy())

            step_count += 1

        prev_gripper = gripper

        # Limit frame rate
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)


def save_episode_to_dataset(
    dataset: LeRobotDataset,
    episode_data: Dict,
    task_name: str,
    camera_names: list,
    source: str,
    success: bool,
) -> None:
    """
    Save episode to LeRobot dataset with source and success labels.

    Args:
        dataset: LeRobotDataset instance
        episode_data: Dict with observations, actions, rewards, etc.
        task_name: Task identifier (e.g., "Stack_Panda")
        camera_names: List of camera names
        source: "policy" or "human"
        success: True/False for task success
    """
    num_frames = len(episode_data["actions"])

    if num_frames == 0:
        print("Warning: No frames to save")
        return

    # Encode source and success as integers
    # source: 0=policy, 1=human
    source_id = 1 if source == "human" else 0
    # success: 0=False, 1=True
    success_flag = 1 if success else 0

    for i in range(num_frames):
        frame = {
            "task": task_name,
            "observation.state": episode_data["observations"][i].astype(np.float32),
            "action": episode_data["actions"][i].astype(np.float32),
            "source": np.array([source_id], dtype=np.int64),  # 0=policy, 1=human
            "success": np.array([success_flag], dtype=np.int64),  # 0=False, 1=True
        }

        # Add camera observations
        for cam_name in camera_names or []:
            cam_key = f"{cam_name}_image"
            if cam_key in episode_data:
                frame[f"observation.images.{cam_name}"] = episode_data[cam_key][i]

        dataset.add_frame(frame)

    dataset.save_episode()
    print(f"✓ Saved episode: source={source}, success={success}, frames={num_frames}")


def main():
    args = parse_args()

    # Check if using mjpython on macOS (required for viewer)
    if platform.system() == "Darwin" and not args.headless:
        import mujoco.viewer

        is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON")
        if not is_mjpython:
            print("=" * 70)
            print("ERROR: On macOS, you must use 'mjpython' instead of 'python'")
            print("=" * 70)
            print("\nPlease run with mjpython:")
            print(f"  mjpython -m sir.teleoperation.robosuite_dagger {' '.join(sys.argv[1:])}")
            print("=" * 70)
            sys.exit(1)

    print("=" * 70)
    print("DAgger Data Collection - Policy Rollout + Human Correction")
    print("=" * 70)
    print(f"W&B Artifact: {args.wandb_artifact}")
    print(f"Environment: {args.env} ({args.robot})")
    print(f"Dataset: {args.dataset_name}")
    print(f"Device: {args.device}")
    print("=" * 70)
    print()

    # Initialize W&B
    print("Initializing W&B...")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="dagger-collection",
        config=vars(args),
    )
    print("✓ W&B initialized")
    print()

    # Load policy from W&B artifact
    print(f"Loading policy from W&B artifact: {args.wandb_artifact}")
    policy, preprocessor, postprocessor = load_policy_from_wandb(
        artifact_identifier=args.wandb_artifact,
        device=args.device,
    )
    print("✓ Policy loaded successfully")
    print()

    # Extract camera configuration from policy
    policy_cameras, policy_height, policy_width = extract_camera_config_from_policy(policy)

    # Parse user-provided cameras
    camera_names = [cam.strip() for cam in args.cameras.split(",")]

    # Use policy's camera config if not explicitly provided
    camera_height = args.camera_height or policy_height
    camera_width = args.camera_width or policy_width

    if camera_height is None or camera_width is None:
        print("ERROR: Could not determine camera resolution")
        print("Please specify --camera-height and --camera-width")
        sys.exit(1)

    print(f"Camera configuration:")
    print(f"  Cameras: {camera_names}")
    print(f"  Resolution: {camera_width}x{camera_height}")
    print()

    # Get action stats for denormalization from preprocessor
    # The preprocessor is a pipeline with multiple steps; stats are in the NormalizerProcessorStep
    action_stats = None
    for step in preprocessor.steps:
        if hasattr(step, "stats") and "action" in step.stats:
            action_stats = step.stats["action"]
            break

    if action_stats is None:
        print("ERROR: No action statistics found in preprocessor")
        print("This might indicate the checkpoint was not saved properly")
        sys.exit(1)

    print(f"✓ Action statistics loaded (mean shape: {action_stats['mean'].shape})")
    print()

    # Initialize SpaceMouse
    print("Initializing SpaceMouse...")
    spacemouse = RobosuiteSpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    print("✓ SpaceMouse initialized")
    print()

    # Initialize keyboard listener
    print("Initializing keyboard listener...")
    kbd_listener = KeyboardListener()
    print("✓ Keyboard listener initialized")
    print()

    # Setup dataset
    dataset_path = Path(args.dataset_path) / args.dataset_name
    dataset = None
    saved_policy_count = 0
    saved_human_count = 0

    print(f"Dataset path: {dataset_path}")
    print()

    print("Setup complete! Starting DAgger data collection...")
    print("=" * 70)

    try:
        episode_count = 0
        env = None

        while True:
            episode_count += 1
            print(f"\n{'='*70}")
            print(f"Episode {episode_count}")
            print(f"{'='*70}")

            # Create fresh environment
            if env is not None:
                env.close()

            env = create_env(
                env_name=args.env,
                robot_name=args.robot,
                camera_names=camera_names,
                camera_height=camera_height,
                camera_width=camera_width,
                controller=args.controller,
                config=args.config,
                has_renderer=not args.headless,
                visual_aids=args.visual_aids,
            )

            # Phase 1: Policy rollout
            policy_data, intervention, task_success = policy_rollout_episode(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                action_stats=action_stats,
                kbd_listener=kbd_listener,
                camera_names=camera_names,
                device=args.device,
                max_steps=args.max_steps,
                max_fr=args.max_fr,
                has_renderer=not args.headless,
                auto_save_on_success=args.auto_save_on_success,
            )

            # Initialize dataset on first save
            task_name = f"{args.env}_{args.robot}"

            if dataset is None and policy_data is not None:
                if dataset_path.exists():
                    print(f"Loading existing dataset from {dataset_path}")
                    dataset = LeRobotDataset(
                        repo_id=args.dataset_name,
                        root=str(dataset_path),
                    )
                    print(f"✓ Loaded existing dataset with {dataset.num_episodes} episodes")
                else:
                    print(f"Creating new dataset at {dataset_path}")
                    obs_shape = policy_data["observations"][0].shape[0]
                    action_shape = policy_data["actions"][0].shape[0]

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
                        # Source: 0=policy, 1=human (encoded as 1D array for compatibility)
                        "source": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": ["source_id"],
                        },
                        # Success: 0=False, 1=True (encoded as 1D array for compatibility)
                        "success": {
                            "dtype": "int64",
                            "shape": (1,),
                            "names": ["success_flag"],
                        },
                    }

                    # Add camera features
                    for cam_name in camera_names:
                        cam_key = f"{cam_name}_image"
                        if cam_key in policy_data:
                            img_shape = policy_data[cam_key][0].shape
                            features[f"observation.images.{cam_name}"] = {
                                "dtype": "video",
                                "shape": img_shape,
                                "names": ["height", "width", "channels"],
                            }

                    dataset = LeRobotDataset.create(
                        repo_id=args.dataset_name,
                        fps=20,
                        root=str(dataset_path),
                        robot_type=args.robot.lower(),
                        features=features,
                    )
                    print(f"✓ Dataset created at {dataset_path}")

            # Save policy trajectory
            if policy_data is not None and len(policy_data["actions"]) > 0:
                save_episode_to_dataset(
                    dataset=dataset,
                    episode_data=policy_data,
                    task_name=task_name,
                    camera_names=camera_names,
                    source="policy",
                    success=task_success,
                )
                saved_policy_count += 1

            # Phase 2: Human correction (if intervention was triggered)
            if intervention:
                human_data, correction_success = spacemouse_correction_episode(
                    env=env,
                    device=spacemouse,
                    kbd_listener=kbd_listener,
                    camera_names=camera_names,
                    max_fr=args.max_fr,
                    has_renderer=not args.headless,
                    auto_save_on_success=args.auto_save_on_success,
                )

                # Save human correction if marked as success
                if correction_success and human_data is not None and len(human_data["actions"]) > 0:
                    save_episode_to_dataset(
                        dataset=dataset,
                        episode_data=human_data,
                        task_name=task_name,
                        camera_names=camera_names,
                        source="human",
                        success=True,
                    )
                    saved_human_count += 1

            # Summary
            print(f"\nEpisode {episode_count} Summary:")
            print(f"  Policy trajectories saved: {saved_policy_count}")
            print(f"  Human corrections saved: {saved_human_count}")
            print(f"  Total episodes in dataset: {dataset.num_episodes if dataset else 0}")
            print()

            # Wait for confirmation before next episode using keyboard listener
            print("Press 'n' to start NEXT episode, or Ctrl+C to quit...")
            while True:
                key = kbd_listener.read_key()
                if key == "n":
                    break
                time.sleep(0.1)  # Small delay to avoid busy waiting

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("\nCleaning up...")
        if dataset is not None:
            print(f"Finalizing dataset...")
            print(f"  Policy trajectories: {saved_policy_count}")
            print(f"  Human corrections: {saved_human_count}")
            dataset.finalize()
        print("Stopping keyboard listener...")
        kbd_listener.close()
        print("Closing SpaceMouse...")
        spacemouse.close()
        if env is not None:
            print("Closing environment...")
            env.close()
        if wandb.run:
            wandb.finish()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
