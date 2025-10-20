"""IK control for Franka Panda robot using Mink.

This module provides inverse kinematics functionality for the Franka Panda robot
in the furniture assembly environment.
"""

import mujoco
import numpy as np
from pathlib import Path
import mink


class FrankaIKController:
    """IK controller for Franka Panda robot using Mink."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize the IK controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

        # Find the end-effector body (hand frame)
        self.ee_body_name = "hand"
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        # Get joint IDs for the 7-DOF arm (excluding gripper)
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]

        # Get actuator IDs corresponding to these joints
        self.actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            for i in range(1, 8)
        ]

        # Create Mink configuration
        self.configuration = mink.Configuration(model)

        # Set up IK tasks
        # Position task for end-effector
        self.position_task = mink.FrameTask(
            frame_name=self.ee_body_name,
            frame_type="body",
            position_cost=1.0,  # High importance
            orientation_cost=1.0,  # High importance
            lm_damping=1.0,  # Levenberg-Marquardt damping
        )

        # Posture task to regularize joint positions
        # Define a reasonable home configuration for the Franka Panda
        # This configuration respects joint limits and is a common rest pose
        self.home_configuration = np.array([
            0.0,      # joint1
            -0.785,   # joint2: -45 degrees
            0.0,      # joint3
            -2.356,   # joint4: -135 degrees (must be in range [-3.0718, -0.0698])
            0.0,      # joint5
            1.571,    # joint6: 90 degrees
            0.785,    # joint7: 45 degrees
        ])

        self.posture_task = mink.PostureTask(
            model=model,
            cost=1e-2,  # Low cost, just for regularization
        )

        # Joint limit constraints
        self.limits = [
            mink.ConfigurationLimit(model=model),
        ]

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose.

        Returns:
            Tuple of (position, quaternion) in world frame
        """
        # Forward kinematics to get end-effector pose
        mujoco.mj_forward(self.model, self.data)

        # Get body pose
        pos = self.data.xpos[self.ee_body_id].copy()
        quat = self.data.xquat[self.ee_body_id].copy()  # [w, x, y, z]

        return pos, quat

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        dt: float = 0.01,
        init_from_home: bool = True,
    ) -> np.ndarray | None:
        """Solve IK for target end-effector pose.

        Args:
            target_pos: Target position [x, y, z] in world frame
            target_quat: Target orientation as quaternion [w, x, y, z].
                        If None, only position is constrained.
            max_iterations: Maximum IK solver iterations
            tolerance: Convergence tolerance in meters
            dt: Integration timestep for IK solver
            init_from_home: If True, initialize from home configuration

        Returns:
            Joint positions (7-DOF) if solution found, None otherwise
        """
        # Initialize from home configuration if requested
        if init_from_home:
            for i, jid in enumerate(self.joint_ids):
                qpos_idx = self.model.jnt_qposadr[jid]
                self.data.qpos[qpos_idx] = self.home_configuration[i]
                self.configuration.q[qpos_idx] = self.home_configuration[i]
            mujoco.mj_forward(self.model, self.data)

        # Set target pose for position task
        self.position_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(target_quat if target_quat is not None else np.array([1, 0, 0, 0])),
                target_pos
            )
        )

        # Set posture target (regularize towards current configuration)
        # Use current configuration as reference to avoid moving furniture
        posture_target = self.configuration.q.copy()
        # Only update the arm joint positions in the posture target
        for i, jid in enumerate(self.joint_ids):
            qpos_idx = self.model.jnt_qposadr[jid]
            posture_target[qpos_idx] = self.home_configuration[i]
        self.posture_task.set_target(posture_target)

        # If no orientation specified, disable orientation cost
        if target_quat is None:
            self.position_task.orientation_cost = 0.0
        else:
            self.position_task.orientation_cost = 1.0

        # Set tasks
        tasks = [self.position_task, self.posture_task]

        # Iterative IK solving
        try:
            for iteration in range(max_iterations):
                # Compute velocity from current configuration to target
                velocity = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=tasks,
                    dt=dt,
                    solver="quadprog",  # Use quadprog solver
                    damping=1e-3,
                    limits=self.limits,
                )

                # Integrate velocity to get new configuration
                self.configuration.integrate_inplace(velocity, dt)

                # Update MuJoCo data with new configuration
                self.data.qpos[:] = self.configuration.q
                mujoco.mj_forward(self.model, self.data)

                # Check convergence
                current_pos, _ = self.get_ee_pose()
                error = np.linalg.norm(current_pos - target_pos)

                if error < tolerance:
                    # Converged!
                    break

            # Extract joint positions for the arm
            q_arm = np.array([
                self.configuration.q[self.model.jnt_qposadr[jid]]
                for jid in self.joint_ids
            ])

            return q_arm

        except Exception as e:
            print(f"IK failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_joint_positions(self, q_arm: np.ndarray):
        """Set arm joint positions in the simulation.

        Args:
            q_arm: 7-DOF joint positions
        """
        for i, jid in enumerate(self.joint_ids):
            qpos_idx = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_idx] = q_arm[i]

        # Forward kinematics to update state
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions.

        Returns:
            7-DOF joint positions
        """
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])


def set_reasonable_initial_pose(model: mujoco.MjModel, data: mujoco.MjData):
    """Set a reasonable initial pose for the Franka robot.

    This function uses IK to position the robot with the end-effector
    above the work table in a good configuration for manipulation.

    Args:
        model: MuJoCo model
        data: MuJoCo data
    """
    # Create IK controller
    ik = FrankaIKController(model, data)

    # Define a reasonable target pose:
    # - Position above the center of the work table
    # - Gripper pointing down
    # Work table is at (0, 0, 0.4), robot mount is at (-0.4, 0, 0.42)
    # Let's position the end-effector at (0.1, 0.2, 0.6) - above and forward of robot base
    target_pos = np.array([0.1, 0.2, 0.6])

    # Gripper pointing down: rotate -90 degrees around Y axis
    # Quaternion for rotation -90° around Y: [cos(-45°), 0, sin(-45°), 0] = [0.7071, 0, -0.7071, 0]
    # But Mujoco uses [w, x, y, z] format
    # For pointing straight down: [0, 0.7071, 0.7071, 0] (rotated around X and Y)
    # Let's use a simple down-pointing orientation
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation around X axis (gripper down)

    print(f"\nSetting initial pose:")
    print(f"  Target position: {target_pos}")
    print(f"  Target orientation (quat): {target_quat}")

    # Solve IK
    q_solution = ik.solve_ik(target_pos, target_quat)

    if q_solution is not None:
        print(f"  IK solution found!")
        print(f"  Joint positions: {q_solution}")

        # Set the joint positions
        ik.set_joint_positions(q_solution)

        # Verify the pose
        actual_pos, actual_quat = ik.get_ee_pose()
        pos_error = np.linalg.norm(actual_pos - target_pos)
        print(f"  Actual position: {actual_pos}")
        print(f"  Position error: {pos_error:.4f} m")

        return True
    else:
        print("  IK solution not found!")
        return False
