# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Robots Learning from their Mistakes** - Research on continual robot learning through RL fine-tuning and online adaptation.

**Research Goal**: Build robots that never stop learning and can learn from their own mistakes. Specifically:
- Why do policies rarely reach 100% success rates?
- What characterizes tasks where Behavioral Cloning (BC) saturates?
- When does the scalability crossover happen (BC → DAgger → RL)?
- How to unify BC + DAgger + RL into a comprehensive framework?
- How to leverage pre-trained VLAs for improved exploration and faster adaptation in RL?

**Current Phase**: Infrastructure setup - SpaceMouse teleoperation complete. Next phases involve:
1. Robot/simulation setup for bimanual manipulation
2. Implement unified BC + RL pipeline with action chunking support
3. VLA fine-tuning experiments (Pi0, OpenVLA + RL methods)
4. Analysis of BC saturation and scaling crossover points

## Environment Setup

### Prerequisites
- macOS (Apple Silicon M1/M2/M3)
- Python 3.11+ managed via micromamba (env: `sir`)
- Homebrew package manager
- 3Dconnexion SpaceMouse device

### Initial Setup
Run the automated setup script for SpaceMouse dependencies:
```bash
./setup_spacemouse.sh
source ~/.zshrc  # Apply environment changes
```

This installs:
- `hidapi` via Homebrew (system dependency)
- `pyspacemouse==1.1.4` (Python library)
- Patched `easyhid` for ARM Mac compatibility
- Configures `DYLD_LIBRARY_PATH` in `.zshrc`

## Package Installation

This project is structured as a proper Python package that can be installed in editable mode:

```bash
# Install in editable mode (development)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

After installation, you can use the package commands from anywhere or run modules directly.

## Common Commands

### Testing SpaceMouse
```bash
# Using installed console script
sir-test-spacemouse

# Or using python -m
python -m sir.tests.test_spacemouse
```
Expected output: Device detection, connection confirmation, real-time 6-DOF values (position: x/y/z, rotation: roll/pitch/yaw, button states). Press Ctrl+C to stop.

### Testing ManiSkill Environment
```bash
# Using installed console script
sir-test-env

# Or using python -m
python -m sir.tests.test_env
```
Expected output: Environment creation, reset confirmation, random actions for 10 steps with reward/termination info.

### Teleoperation

**ManiSkill (no special requirements):**
```bash
# Using installed console script
sir-teleop -e PegInsertionSide-v1

# Or using python -m
python -m sir.teleoperation -e PegInsertionSide-v1

# Adjust control parameters if needed
sir-teleop -e PickCube-v1 --speed 0.2 --rot-speed 0.4
sir-teleop -e StackCube-v1 --stiffness 3000 --damping 300
```

**Robosuite (IMPORTANT - use mjpython on macOS):**
```bash
# macOS - MUST use mjpython (not python)
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

# Linux - use python
python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

# With camera observations
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
  --save-images \
  --cameras "agentview,robot0_eye_in_hand"
```

**Why mjpython on macOS?** The MuJoCo viewer requires GUI operations on the main thread. Regular Python runs GUI code on background threads, causing NSWindow crashes. MuJoCo's `mjpython` wrapper ensures GUI operations run on the main thread.

### Manual Environment Setup
If `DYLD_LIBRARY_PATH` is not set:
```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hidapi/0.15.0/lib:$DYLD_LIBRARY_PATH
```
Version may differ - check with `brew --prefix hidapi`

## Architecture & Key Concepts

### SpaceMouse Teleoperation
- **Purpose**: Human-in-the-loop data collection for demonstrations and corrections
- **Library**: `pyspacemouse` - cross-platform, no official 3Dconnexion drivers required
- **Interface**: Raw HID access via `hidapi`
- **Data**: 6-DOF input stream at ~100Hz (10ms intervals)
  - Translation: `state.x`, `state.y`, `state.z`
  - Rotation: `state.roll`, `state.pitch`, `state.yaw`
  - Buttons: `state.buttons` (array)
- **API Pattern**:
  ```python
  pyspacemouse.open()  # Initialize connection
  state = pyspacemouse.read()  # Non-blocking read
  pyspacemouse.close()  # Cleanup
  ```
- **Use Cases**:
  - Collecting demonstration data for BC pre-training
  - Providing corrective interventions during RL rollouts (HiL-SERL, RaC-style)
  - Shared autonomy for safe exploration

### ManiSkill3 Teleoperation Configuration
- **Control Mode**: `pd_ee_target_delta_pose` (default, recommended for teleoperation)
  - Uses target-based control: actions relative to last commanded pose, not actual pose
  - Prevents drift/drooping when holding objects or under load
  - Robot actively maintains commanded position even with zero input
  - Alternative: `pd_ee_delta_pose` (may droop under load)
- **PD Controller Gains** (tuned for responsive teleoperation):
  - Stiffness: 2000 (default 1000) - higher = more responsive to commands
  - Damping: 200 (default 100) - higher = less oscillation
  - Force limit: 200 (default 100) - higher = stronger position holding
- **Frame Mapping**: SpaceMouse → Robot coordinate frames
  - Translation: `-yxz` (negate Y→X, X→Y, Z→Z)
  - Rotation: `-x-yz` (negate X→X, negate Y→Y, Z→Z)
  - Makes controls intuitive: forward=forward, left=left
- **Deadzone**: 0.02 threshold filters sensor noise to prevent unwanted drift
- **Control Speeds**: Translation 0.15 m/s, Rotation 0.3 rad/s (defaults, adjustable)

### Robosuite Teleoperation Configuration
- **SpaceMouse Driver**: Custom implementation using `hidapi` directly (matches Robosuite's original)
  - Device IDs: Vendor `0x256F` (3Dconnexion), Product `0xC635` (SpaceMouse Compact)
  - Different models may need different product IDs
- **Control Scaling**: Matches Robosuite exactly
  - Base scaling: `0.005` for all axes
  - Position multiplier: `125` (0.625 total)
  - Rotation multiplier: `50` (0.25 total)
  - Final clipping to `[-1, 1]` range
- **Coordinate Frame Transform**: Robot expects `[pitch, roll, -yaw]` not `[roll, pitch, yaw]`
  - Reorder: `raw_drot[[1, 0, 2]]`
  - Flip yaw sign: `drot[2] = -drot[2]`
- **Camera Observations**: Configurable camera names, resolution, automatic image saving
  - Common cameras: `agentview` (third-person), `robot0_eye_in_hand` (wrist-mounted)
  - Requires `has_offscreen_renderer=True` and `use_camera_obs=True`
  - **macOS BREAKTHROUGH**: CAN use onscreen viewer + camera obs simultaneously!
  - Fixed Robosuite bug: added CGL backend support (was recognized but not used)
  - CGL (Core OpenGL) contexts are NOT tied to main thread on macOS
  - Script automatically sets `MUJOCO_GL=cgl` on macOS
  - Works with mjpython viewer without conflicts
  - Example images saved to `./robosuite_images/` by default
- **Viewer Limitation**: MuJoCo viewer cannot be reopened → environment recreated for each episode
- **macOS Critical**: MUST use `mjpython` not `python` - GUI operations require main thread

### Environment-Specific Notes
- **ARM Mac SpaceMouse**: Standard `easyhid` doesn't work on Apple Silicon; must use patched version from https://github.com/bglopez/python-easyhid.git
- **Library Loading**: macOS requires `DYLD_LIBRARY_PATH` to locate `hidapi` dynamic library at runtime
- **3Dconnexion Driver Conflicts**: If official drivers are installed, `3DconnexionHelper` process may interfere. Kill with `killall 3DconnexionHelper` if connection fails.
- **macOS MuJoCo Viewer**: NSWindow must be instantiated on main thread → use `mjpython` not `python`

### User Preferences
- **Reproducibility**: Always create automation scripts and documentation for setup/configuration
- **Documentation**: Provide both automated and manual setup paths
- **Shell**: User runs zsh with oh-my-zsh

## Research Directions & Implementation Priorities

### 1. Analysis: BC Saturation Characteristics
**Goal**: Understand when and why BC reaches performance limits
- Scaling experiments: Track BC performance vs. demo count
- Identify task characteristics where BC saturates vs. reaches 100%
- Measure "crossover point" where DAgger/RL becomes beneficial
- Metric: Total trajectories (not samples) to reach target performance

### 2. Method: Unified BC + DAgger + RL Framework
**Goal**: Seamless progression from demos → corrections → autonomous improvement
- Support action chunking (crucial for performance and exploration)
- Handle long-horizon, bimanual manipulation
- Enable both local corrections and global behavior optimization
- Scale naturally with offline demos and online rollouts
- Integrate human corrections when needed (shared autonomy)

### 3. Pre-training → Fine-tuning with VLAs
**Goal**: Leverage large-scale pre-trained models for faster adaptation and better exploration
- Fine-tune VLAs (Pi0, OpenVLA) with RL objectives
- Test hypothesis: stronger pre-training → better exploration
- Test hypothesis: stronger pre-training → faster adaptation
- Apply recent RL methods to VLAs: QC, DPPO, ResFiT/ResiP, ConRFT, Policy Decorator

### 4. Target Task Properties
Focus on tasks that are:
- **Bimanual**: Require coordinated two-arm manipulation
- **Long-horizon**: Multi-step tasks with temporal dependencies
- **Demo-driven**: Start from teleoperated demonstrations
- **Vision-based**: Standard RGB cameras (no special sensors)
- **Parallel-jaw grippers**: Avoid dexterous hand complexity initially

**Example Tasks** (from RaC paper):
- Shirt hanging (deformable object manipulation)
- Bimanual belt assembly (coordination + precision)
- Bimanual kitchen tasks (pick, handover, insert plate in rack)
- Bimanual furniture assembly (long-horizon coordination)

### 5. De-risking Experiments (Ordered by Priority)
1. **Scaling crossover**: Simple task, vary BC pre-training data, measure online trajectories to convergence
2. **Objective unification**: Insertion task, compare BC-only vs. BC+corrections vs. BC+RL+corrections
3. **VLA + RL integration**: Fine-tune Pi0 on simple task, apply QC/DPPO/ResFiT methods

## Technical Approach

### Infrastructure & Tooling
- **Training/Data Framework**: HuggingFace LeRobot (preferred over ManiSkill built-in code)
  - Environment-agnostic dataset format
  - State-of-the-art policy implementations (ACT, Diffusion Policy, VLA fine-tuning)
  - Easy integration with HuggingFace Hub for sharing
  - Better maintained and more widely adopted
- **Simulation**: ManiSkill3 (for environments and physics)
- **Data Collection**: SpaceMouse teleoperation (custom integration)
- **Data Format**: Convert ManiSkill HDF5 → LeRobot format for training

### Action Representation
- **Action chunking** is essential (enables temporal consistency and better exploration)
- Support for both diffusion policies and direct regression
- Timestep-level rewards but chunk-level action optimization

### RL Algorithms to Consider
- **Off-policy methods**: SAC, AWAC, IDQL, CQL (for offline → online)
- **On-policy methods**: PPO, DPPO (for VLA fine-tuning)
- **Residual methods**: ResFiT, ResiP (add residual RL on top of BC policy)
- **Q-learning methods**: QC, CQN (for chunked actions)

### Human-in-the-Loop Integration
- Adaptive shared autonomy during rollouts
- Intervention recording as high-quality corrections
- DAgger-style query mechanisms
- Safe exploration via human monitoring

## Troubleshooting

### SpaceMouse Not Found
1. Verify device is plugged in (check USB connection)
2. Grant Input Monitoring permissions: System Settings → Privacy & Security → Input Monitoring → Add Terminal/IDE
3. Kill conflicting drivers: `killall 3DconnexionHelper`
4. Verify hidapi: `ls -la $(brew --prefix hidapi)/lib/`

### Library Loading Errors
- Symptom: `AttributeError: function/symbol 'hid_enumerate' not found`
- Cause: `DYLD_LIBRARY_PATH` not set or incorrect
- Fix: Re-run `setup_spacemouse.sh` or manually set environment variable

## Resources
- [PySpaceMouse Documentation](https://spacemouse.kubaandrysek.cz/)
- [PySpaceMouse GitHub](https://github.com/JakubAndrysek/PySpaceMouse)
- Detailed setup guide: `SETUP_SPACEMOUSE.md`
- Development notes: `.claude/notes.md`
