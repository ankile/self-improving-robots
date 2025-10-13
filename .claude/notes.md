# Claude Development Notes

This file contains context and notes for future Claude sessions working on this project.

## Project Overview

**Repository**: `self-improving-robots`
**Owner**: larsankile (REAL Lab)
**Research Focus**: "Robots Learning from their Mistakes" - continual robot learning through RL fine-tuning

**Core Research Questions**:
1. Why do robot policies rarely reach 100% success rates?
2. What characterizes tasks where BC saturates vs. reaches 100%?
3. When is the scaling crossover point from BC → DAgger → RL?
4. How to unify BC + DAgger + RL into comprehensive framework?
5. Does stronger pre-training lead to better exploration in RL?
6. Does stronger pre-training lead to faster adaptation in RL?

**Research Thesis**: Robots should never stop learning. Current paradigm of "train once, deploy" is insufficient. Need systems that continuously improve through online interaction, learning from their own mistakes with minimal human intervention.

## Current State (2025-10-13)

### What's Been Done

1. **SpaceMouse Integration - COMPLETE ✅**
   - Set up `pyspacemouse` library for macOS (Apple Silicon)
   - Installed system dependencies: `hidapi` via Homebrew
   - Configured environment: `DYLD_LIBRARY_PATH` in `.zshrc`
   - Created test script: `test_spacemouse.py`
   - Created setup automation: `setup_spacemouse.sh`
   - Tested successfully with SpaceMouse Compact
   - Purpose: Teleoperation for demo collection and human-in-the-loop corrections

2. **Documentation - COMPLETE ✅**
   - README.md: Research overview and quick start
   - CLAUDE.md: Detailed technical guidance
   - SETUP_SPACEMOUSE.md: Setup instructions
   - This file: Development context

### Technical Details

**Environment:**
- macOS (Darwin 25.1.0)
- Apple Silicon (arm64)
- Python 3.11
- Micromamba for environment management (env: `sir`)

**SpaceMouse Setup:**
- Library: `pyspacemouse==1.1.4`
- Dependency: Patched `easyhid` from https://github.com/bglopez/python-easyhid.git
- System lib: `hidapi` (version 0.15.0) at `/opt/homebrew/Cellar/hidapi/0.15.0/lib`
- Environment: `DYLD_LIBRARY_PATH` configured in user's `.zshrc`
- Device tested: SpaceMouse Compact (successfully detected and read)
- Use case: 6-DOF teleoperation for bimanual manipulation demos

## What's Next

### Immediate Priorities (De-risking Experiments)

1. **Scaling crossover analysis**
   - Simple BC + RL pipeline
   - Pre-train with varying data amounts
   - Track online trajectories (not samples) to target performance
   - Goal: Characterize when RL becomes beneficial

2. **Unifying objectives**
   - Simple insertion task
   - Compare: BC-only vs. BC+corrections vs. BC+rollouts+corrections vs. BC+RL+rollouts+corrections
   - Goal: Understand value of different data types

3. **VLA + RL integration**
   - Fine-tune Pi0 on simple task
   - Apply recent RL methods: QC, DPPO, ResFiT/ResiP
   - Goal: Test feasibility of VLA fine-tuning with RL

### Research Directions

1. **Analysis**: BC saturation characteristics
   - When/why does BC saturate?
   - Task properties that predict BC success
   - Scaling laws for demo → performance

2. **Methods**: Unified framework
   - BC + DAgger + RL in single pipeline
   - Action chunking support (crucial for exploration)
   - Bimanual manipulation
   - Human-in-the-loop corrections (shared autonomy)
   - Works with any base model (direct, diffusion, VLA)

3. **Pre-training → Fine-tuning**
   - Leverage VLAs (Pi0, OpenVLA, TRI LBM)
   - Hypothesis: better pre-training → better exploration
   - Hypothesis: better pre-training → faster adaptation
   - Test on real robot tasks

### Target Tasks (Bimanual, Long-Horizon, Demo-driven)

- Shirt hanging (deformable object manipulation)
- Bimanual belt assembly (coordination + precision)
- Kitchen: pick, handover, insert plate in rack
- Furniture assembly (long-horizon coordination)

**Requirements**:
- Teleoperable with SpaceMouse
- Parallel-jaw grippers (no dexterous hands yet)
- Vision-based (standard RGB cameras)
- Some precision required (not trivial)

### Infrastructure Needs

- **Robot/simulation environment**
  - Bimanual arms with parallel-jaw grippers
  - RGB camera(s) for vision
  - Real robot OR high-fidelity sim (ask user preference)
- **BC baseline** with action chunking
- **RL fine-tuning pipeline** (SAC, PPO, residual methods)
- **Data collection system** (demos, rollouts, corrections)
- **Evaluation metrics** and logging

## Technical Decisions Made

1. **SpaceMouse for teleoperation**: Most actively maintained library (pyspacemouse), cross-platform, no driver dependencies
2. **ARM-specific patches**: Required patched easyhid for Apple Silicon compatibility
3. **Environment variable approach**: Added to zshrc for persistence vs. wrapper scripts
4. **Automated setup script**: Created for reproducibility and onboarding
5. **Focus on action chunking**: Critical for performance and exploration in manipulation tasks
6. **Demo-driven approach**: Start with BC, then add RL (vs. RL from scratch)
7. **Human-in-the-loop**: Design for corrections and shared autonomy from the start
8. **Vision-based policies**: Standard RGB cameras (avoid force/tactile sensors initially)
9. **Bimanual manipulation**: More interesting research problems than single-arm

## Key Insights from Literature Review (from presentation)

**BC Limitations**:
- Many tasks don't reach 100% with BC alone (Aloha Unleashed, ResiP, DexMimicGen, Self-Improving EFM)
- Reasons: not enough demos, wrong architecture, partial observability, distribution shift, train/test mismatch
- Even 1000s of demos can saturate at 70-80% (ResiP: one_leg task ~75% with 10k demos)

**Scaling Crossover**:
- BC has diminishing returns at some point
- RL becomes more efficient beyond crossover
- Need to characterize this crossover systematically
- Alternative: unified approach with RL objective from start (adaptive shared autonomy)

**Pre-training Exploration** (Russ Tedrake comment):
- "When my boy Russ is here, I listen" - community recognizes importance
- Stronger pre-training likely enables more structured exploration
- Not fully explored in RL context yet (DPPO, QC show promise)

**Method Requirements** (from recent work):
- Long horizons, sparse rewards, sample efficiency
- Works with any base model (VLA, diffusion, direct)
- Safe exploration, human corrections when needed
- Examples exist separately, but not all combined

## Known Issues / Limitations

- SpaceMouse setup complete and working
- No robot/sim environment yet
- No BC/RL pipeline implemented yet

## Tips for Future Claude Sessions

1. **Research context**: This is about continual learning and RL fine-tuning, NOT just generic "self-improving robots"
2. **SpaceMouse**: Teleoperation tool for data collection - already set up and working
3. **Focus**: Bimanual, long-horizon, demo-driven manipulation tasks
4. **Methods**: Unifying BC + DAgger + RL with action chunking and VLAs
5. **User preferences**: Clear reproducible setup, automation scripts, thorough documentation

## Questions to Ask User (if needed)

**Robot Platform**:
- Do you have access to bimanual robot setup? Which robots/sim?
- Preference: Real robot vs. simulation for initial experiments?
- What cameras/sensors are available?

**Existing Codebases**:
- Any existing BC/RL codebase to build on?
- Preference for which VLA to start with (Pi0, OpenVLA, other)?
- Any existing task environments set up?

**Research Priorities**:
- Which research direction is highest priority: Analysis, Methods, or VLA fine-tuning?
- Problem-first or method-first approach?
- Timeline and milestones?

**Collaboration**:
- Who else is on the team? Roles?
- Compute resources available (GPUs, clusters)?
- Lab resources (robots, space, etc.)?

## References & Related Work

See README.md for comprehensive list. Key papers:
- **Early RL**: QT-Opt, SERL, IBRL, CQN
- **Action Chunking + RL**: ResFiT, ResiP, DPPO, EXPO
- **VLA Fine-tuning**: Pi0, OpenVLA, ConRFT, Policy Decorator, Self-Improving EFM
- **Human-in-the-Loop**: HiL-SERL, RaC, RoboCopilot

## Resources

- PySpaceMouse docs: https://spacemouse.kubaandrysek.cz/
- PySpaceMouse GitHub: https://github.com/JakubAndrysek/PySpaceMouse
- Research presentation: `/Users/larsankile/Downloads/Project-dicussion.pdf`

---

*Last updated: 2025-10-13*
*Session: Updated with research presentation context*
