# Robots Learning from their Mistakes

Research on continual robot learning: analyzing BC limitations, unified BC+RL frameworks, and VLA fine-tuning for improved exploration and adaptation.

## Research Overview

**Core Question**: Why do robot policies rarely reach 100% success rates, and how can robots continuously learn from their mistakes?

**Key Research Directions**:
1. **Analysis**: Characterize when/why Behavioral Cloning (BC) saturates and identify the scaling crossover point to DAgger/RL
2. **Methods**: Develop unified BC + DAgger + RL framework supporting action chunking, bimanual control, and human-in-the-loop corrections
3. **Pre-training → Fine-tuning**: Explore how pre-trained VLAs (Pi0, OpenVLA) affect RL exploration and adaptation speed

## Current Status

**Phase**: Infrastructure setup

### Implemented
- ✅ SpaceMouse teleoperation for demos/corrections
  - 6-DOF input: position (x, y, z) + rotation (roll, pitch, yaw)
  - Button state monitoring
  - ~100Hz sampling rate
  - Human-in-the-loop data collection ready
- ✅ ManiSkill3 integration
  - SpaceMouse teleoperation script with target-based control
  - Works with any ManiSkill environment
  - Improved PD controller gains (stiffness=2000, damping=200, force_limit=200)
  - Target delta control prevents drooping when holding objects
  - Intuitive frame mapping: SpaceMouse axes aligned to robot frame
  - Deadzone filtering to prevent drift
- ✅ LeRobotDataset integration
  - Save demonstrations directly to LeRobot format during teleoperation
  - Mark episodes as success/failure with keyboard shortcuts
  - Compatible with HuggingFace LeRobot training pipelines
  - Collects observations, actions, rewards, and success labels

### Next Steps
- [x] Add trajectory recording functionality with LeRobotDataset integration
- [ ] Test teleoperation on bimanual tasks (TwoRobotStackCube-v1)
- [ ] BC baseline implementation with action chunking (using LeRobot)
- [ ] RL fine-tuning pipeline (SAC, PPO, or residual methods)
- [ ] VLA integration (Pi0/OpenVLA fine-tuning experiments)
- [ ] Analysis pipeline for BC saturation experiments

## Quick Start

### Installation

**1. Run SpaceMouse setup** (automated, recommended):
```bash
./setup_spacemouse.sh
source ~/.zshrc  # or restart terminal
```

**2. Install the package in editable mode**:
```bash
pip install -e .
```

This installs the `sir` package that can be used via `python -m` commands.

For manual SpaceMouse setup, see [SETUP_SPACEMOUSE.md](SETUP_SPACEMOUSE.md).

### Testing

**Test SpaceMouse connection**:
```bash
python -m sir.tests.test_spacemouse
```

**Test ManiSkill environment**:
```bash
python -m sir.tests.test_env
```

### SpaceMouse Teleoperation

**Basic usage**:
```bash
python -m sir.teleoperation -e PegInsertionSide-v1
```

**Other available tasks**:
```bash
python -m sir.teleoperation -e PickCube-v1
python -m sir.teleoperation -e StackCube-v1
python -m sir.teleoperation -e PushT-v1
```

**Adjust control parameters**:
```bash
# Adjust control speed
python -m sir.teleoperation -e PegInsertionSide-v1 --speed 0.2 --rot-speed 0.4

# Adjust PD controller gains for different responsiveness
python -m sir.teleoperation -e PegInsertionSide-v1 --stiffness 3000 --damping 300

# Save demonstration data to LeRobotDataset
python -m sir.teleoperation -e PegInsertionSide-v1 --save-data --dataset-path ./data --dataset-name my_demos
```

**Controls during teleoperation**:
- Move SpaceMouse: Control robot end-effector (6-DOF)
- Left button: Toggle gripper open/close
- `1` key: Mark episode as SUCCESS and reset (saves to dataset if `--save-data` enabled)
- `0` key: Mark episode as FAILURE and reset (saves to dataset if `--save-data` enabled)
- Ctrl+C: Quit

**Control Modes**:
- `pd_ee_target_delta_pose` (default): Target-based control, prevents drooping when holding objects
- `pd_ee_delta_pose`: Direct delta control (may droop under load)
- `pd_ee_target_delta_pos`: Target-based position-only control (3-DOF)
- `pd_ee_delta_pos`: Direct position-only delta control (3-DOF)

**Data Collection**: Episodes can be saved to LeRobotDataset format by enabling `--save-data`. Press `1` to mark success or `0` to mark failure, and the episode will be saved locally for training.

## Requirements

- macOS (tested on Apple Silicon M1/M2/M3)
- Python 3.11+
- Homebrew
- 3Dconnexion SpaceMouse device

## Target Tasks

Focus on manipulation tasks with these properties:
- **Bimanual coordination**: Two-arm tasks requiring synchronization
- **Long-horizon**: Multi-step tasks with temporal dependencies
- **Vision-based**: Standard RGB cameras (no force/tactile sensors initially)
- **Parallel-jaw grippers**: Avoid dexterous manipulation complexity

**Example Tasks** (inspired by recent work):
- Shirt hanging (deformable objects)
- Bimanual belt assembly (coordination + precision)
- Kitchen tasks: pick, handover, insert plate in rack
- Furniture assembly (long-horizon bimanual)

## Method Requirements

Develop methods that handle:
- ✅ Long horizons (temporal credit assignment)
- ✅ Sparse rewards (exploration challenges)
- ✅ Sample efficiency (real robot constraints)
- ✅ Action chunking (temporal consistency)
- ✅ Any base model architecture (VLA, diffusion, direct)
- ✅ Safe exploration (human monitoring/intervention)
- ✅ Human corrections when needed (shared autonomy)

## Project Structure

```
.
├── README.md                      # This file
├── CLAUDE.md                      # Detailed guidance for Claude Code
├── SETUP_SPACEMOUSE.md           # SpaceMouse setup guide
├── pyproject.toml                # Package configuration and dependencies
├── setup.py                       # Setup script for pip install -e .
├── setup_spacemouse.sh           # Automated SpaceMouse setup script
├── requirements-spacemouse.txt   # SpaceMouse-specific dependencies
├── sir/                          # Main package directory
│   ├── __init__.py
│   ├── teleoperation/            # Teleoperation modules
│   │   ├── __init__.py
│   │   ├── __main__.py           # Entry point for python -m sir.teleoperation
│   │   ├── spacemouse_teleop.py  # Main teleoperation script
│   │   ├── utils.py              # Utility functions (deadzone, axis mapping, keyboard)
│   │   └── robot_config.py       # Robot configuration utilities
│   └── tests/                    # Test scripts
│       ├── __init__.py
│       ├── test_spacemouse.py    # SpaceMouse test script
│       ├── test_spacemouse/      # Entry point for python -m
│       ├── test_maniskill_env.py # ManiSkill environment test
│       └── test_env/             # Entry point for python -m
└── .claude/
    └── notes.md                  # Development notes
```

The package can be installed with `pip install -e .` and accessed via `python -m` commands:
- `python -m sir.teleoperation` - SpaceMouse teleoperation
- `python -m sir.tests.test_spacemouse` - Test SpaceMouse connection
- `python -m sir.tests.test_env` - Test ManiSkill environment

## Hardware

**Teleoperation Device** (for demo collection):
- Tested: SpaceMouse Compact
- Supported: SpaceNavigator, SpaceMouse Pro/Wireless, SpacePilot

**Robot Platform** (TBD):
- Need bimanual setup with parallel-jaw grippers
- RGB camera(s) for vision-based policy
- Real robot or high-fidelity simulation

## Related Work & Inspiration

**Early Real-World RL**:
- QT-Opt, SERL, CQN (sample-efficient RL from scratch)
- IBRL (imitation bootstrapped RL)

**Action Chunking + RL**:
- ResFiT, ResiP (residual RL on BC policies)
- DPPO (diffusion policy optimization)
- EXPO (stable RL with expressive policies)

**VLA Fine-tuning**:
- Pi0, OpenVLA (large-scale pre-trained models)
- ConRFT, Policy Decorator (RL fine-tuning of VLAs)
- Self-Improving Foundation Models

**Human-in-the-Loop**:
- HiL-SERL (RL with human corrections)
- RaC (recovery and correction for long-horizon)
- RoboCopilot (interactive imitation learning)

## Resources

- [PySpaceMouse Documentation](https://spacemouse.kubaandrysek.cz/)
- [PySpaceMouse GitHub](https://github.com/JakubAndrysek/PySpaceMouse)
- [CLAUDE.md](CLAUDE.md) - Detailed technical guidance
- [.claude/notes.md](.claude/notes.md) - Development context

## License

TBD
