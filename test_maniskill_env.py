#!/usr/bin/env python3
"""
Quick test script to verify ManiSkill environment is working.
This script will load an environment and take random actions for a few steps.
"""

import gymnasium as gym
import numpy as np
import mani_skill.envs  # Register ManiSkill environments

def test_env(env_id="PegInsertionSide-v1", num_steps=10):
    print(f"Testing environment: {env_id}")
    print("=" * 60)

    # Create environment
    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        robot_uids="panda"
    )

    print(f"✓ Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Reset environment
    obs, info = env.reset(seed=0)
    print(f"✓ Environment reset successfully!")
    print(f"Info: {info}")
    print()

    # Take random actions
    print(f"Taking {num_steps} random actions...")
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Convert tensors to floats for printing
        reward_val = reward.item() if hasattr(reward, 'item') else reward
        term_val = terminated.item() if hasattr(terminated, 'item') else terminated
        trunc_val = truncated.item() if hasattr(truncated, 'item') else truncated
        print(f"Step {step + 1}: reward={reward_val:.4f}, terminated={term_val}, truncated={trunc_val}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break

    print()
    print("✓ Test completed successfully!")

    # Clean up
    env.close()

if __name__ == "__main__":
    try:
        test_env("PegInsertionSide-v1", num_steps=10)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
