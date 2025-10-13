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

### Next Steps
- [ ] Robot/simulation environment setup (bimanual manipulation)
- [ ] BC baseline implementation with action chunking
- [ ] RL fine-tuning pipeline (SAC, PPO, or residual methods)
- [ ] VLA integration (Pi0/OpenVLA fine-tuning experiments)
- [ ] Analysis pipeline for BC saturation experiments

## Quick Start

### SpaceMouse Setup

1. **Automated setup** (recommended):
   ```bash
   ./setup_spacemouse.sh
   source ~/.zshrc  # or restart terminal
   ```

2. **Test SpaceMouse connection**:
   ```bash
   python test_spacemouse.py
   ```

### Manual Setup

See [SETUP_SPACEMOUSE.md](SETUP_SPACEMOUSE.md) for detailed instructions.

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
├── setup_spacemouse.sh           # Automated setup script
├── requirements-spacemouse.txt   # Python dependencies
├── test_spacemouse.py            # SpaceMouse test/demo script
└── .claude/
    └── notes.md                  # Development notes
```

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
