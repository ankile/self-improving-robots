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

## Common Commands

### Testing SpaceMouse
```bash
python test_spacemouse.py
```
Expected output: Device detection, connection confirmation, real-time 6-DOF values (position: x/y/z, rotation: roll/pitch/yaw, button states). Press Ctrl+C to stop.

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

### Environment-Specific Notes
- **ARM Mac Requirement**: Standard `easyhid` doesn't work on Apple Silicon; must use patched version from https://github.com/bglopez/python-easyhid.git
- **Library Loading**: macOS requires `DYLD_LIBRARY_PATH` to locate `hidapi` dynamic library at runtime
- **3Dconnexion Driver Conflicts**: If official drivers are installed, `3DconnexionHelper` process may interfere. Kill with `killall 3DconnexionHelper` if connection fails.

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
