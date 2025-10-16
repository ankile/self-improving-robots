"""Robot configuration utilities for teleoperation."""

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.registration import register_agent


def create_custom_panda(stiffness: float, damping: float, force_limit: float):
    """
    Create a custom Panda robot with tuned control parameters.

    Args:
        stiffness: PD controller stiffness (higher = more responsive)
        damping: PD controller damping (higher = less oscillation)
        force_limit: Maximum force for PD controller

    Returns:
        Custom Panda robot class with specified parameters
    """
    @register_agent()
    class CustomPanda(Panda):
        uid = "panda_custom_teleop"
        arm_stiffness = stiffness
        arm_damping = damping
        arm_force_limit = force_limit

    return CustomPanda
