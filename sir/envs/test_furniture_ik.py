"""Test IK control for Franka robot in furniture assembly environment."""

import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np

from sir.envs.franka_ik import FrankaIKController, set_reasonable_initial_pose


def main():
    """Test IK control by setting different target poses."""
    # Load the furniture assembly model with Franka
    xml_path = Path(__file__).parent / "furniture_assembly_with_franka.xml"

    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"✓ Model loaded: {model.nq} DOFs, {model.nbody} bodies")
    print(f"✓ Actuators: {model.nu}")

    # Initialize IK controller
    ik = FrankaIKController(model, data)

    # Get initial end-effector pose
    print("\n" + "=" * 60)
    print("INITIAL CONFIGURATION")
    print("=" * 60)
    initial_pos, initial_quat = ik.get_ee_pose()
    initial_joints = ik.get_joint_positions()
    print(f"Initial EE position: {initial_pos}")
    print(f"Initial EE orientation (quat): {initial_quat}")
    print(f"Initial joint positions: {initial_joints}")

    # Set a reasonable initial pose
    print("\n" + "=" * 60)
    print("SETTING REASONABLE INITIAL POSE")
    print("=" * 60)
    success = set_reasonable_initial_pose(model, data)

    if success:
        print("\n✓ Successfully set initial pose!")

        # Get new pose
        new_pos, new_quat = ik.get_ee_pose()
        new_joints = ik.get_joint_positions()
        print(f"\nNew EE position: {new_pos}")
        print(f"New EE orientation (quat): {new_quat}")
        print(f"New joint positions: {new_joints}")
    else:
        print("\n✗ Failed to set initial pose")

    # Try a few more target positions
    print("\n" + "=" * 60)
    print("TESTING ADDITIONAL TARGET POSES")
    print("=" * 60)

    test_targets = [
        {
            "name": "Center of table",
            "pos": np.array([0.0, 0.0, 0.55]),
            "quat": np.array([0.0, 1.0, 0.0, 0.0]),  # Gripper down
        },
        {
            "name": "Left side of table",
            "pos": np.array([0.2, 0.3, 0.55]),
            "quat": np.array([0.0, 1.0, 0.0, 0.0]),  # Gripper down
        },
        {
            "name": "Right side of table",
            "pos": np.array([0.2, -0.3, 0.55]),
            "quat": np.array([0.0, 1.0, 0.0, 0.0]),  # Gripper down
        },
    ]

    for i, target in enumerate(test_targets, 1):
        print(f"\n--- Test {i}: {target['name']} ---")
        print(f"Target position: {target['pos']}")

        q_solution = ik.solve_ik(target["pos"], target["quat"])

        if q_solution is not None:
            ik.set_joint_positions(q_solution)
            actual_pos, _ = ik.get_ee_pose()
            error = np.linalg.norm(actual_pos - target["pos"])
            print(f"✓ IK solved! Position error: {error:.4f} m")
            print(f"  Actual position: {actual_pos}")
        else:
            print("✗ IK failed for this target")

    # Launch viewer with final pose
    print("\n" + "=" * 60)
    print("LAUNCHING VIEWER")
    print("=" * 60)
    print("Press Ctrl+C to close.")
    print("\nThe robot should be positioned in a good configuration for manipulation.")
    print("You can interact with the viewer to inspect the pose.")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
