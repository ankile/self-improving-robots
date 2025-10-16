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

## Current State (2025-10-13, Session 4)

### What's Been Done

1. **SpaceMouse Integration - COMPLETE ✅** (Session 1)
   - Set up `pyspacemouse` library for macOS (Apple Silicon)
   - Installed system dependencies: `hidapi` via Homebrew
   - Configured environment: `DYLD_LIBRARY_PATH` in `.zshrc`
   - Created test script: `test_spacemouse.py`
   - Created setup automation: `setup_spacemouse.sh`
   - Tested successfully with SpaceMouse Compact
   - Purpose: Teleoperation for demo collection and human-in-the-loop corrections

2. **ManiSkill3 Integration - COMPLETE ✅** (Session 2)
   - Explored ManiSkill3 codebase at `/Users/larsankile/code/maniskill`
   - Identified bimanual tasks: `TwoRobotPickCube-v1`, `TwoRobotStackCube-v1`
   - Reviewed existing baselines: BC, Diffusion Policy, ACT, PPO, SAC, TD-MPC2
   - Reviewed teleoperation system (click+drag based)
   - Created SpaceMouse teleoperation script: `spacemouse_teleop.py`
   - Created test script: `test_maniskill_env.py`
   - Created comprehensive guide: `SPACEMOUSE_TELEOP.md`

3. **Teleoperation Improvements - COMPLETE ✅** (Sessions 3-4)
   - Increased PD controller gains for responsiveness (stiffness 2000, damping 200, force_limit 200)
   - Implemented axis remapping system for intuitive control
   - Found working axis mappings: `-yxz` (translation), `-x-yz` (rotation)
   - Added deadzone filtering (0.02 threshold) to prevent drift
   - **Fixed drooping issue**: Switched to `pd_ee_target_delta_pose` control mode
     - Target-based control prevents accumulation of drift errors
     - Robot maintains commanded position even when holding objects
   - Removed trajectory recording temporarily (will re-implement with LeRobot)
   - Teleoperation fully tested and working on PegInsertionSide-v1

4. **Documentation - COMPLETE ✅** (Sessions 1-4)
   - README.md: Research overview and quick start (updated with teleoperation status)
   - CLAUDE.md: Detailed technical guidance (updated with control mode details)
   - SETUP_SPACEMOUSE.md: Setup instructions
   - SPACEMOUSE_TELEOP.md: Teleoperation guide (deprecated, info moved to README)
   - This file: Development context

### Technical Details

**Environment:**
- macOS (Darwin 25.1.0)
- Apple Silicon (arm64)
- Python 3.11 (at `/opt/homebrew/bin/python3`)
- Micromamba for environment management (env: `sir`)
- **IMPORTANT**: Always use `python` command (NOT `python3`) to use the activated micromamba environment

**SpaceMouse Setup:**
- Library: `pyspacemouse==1.1.4`
- Dependency: Patched `easyhid` from https://github.com/bglopez/python-easyhid.git
- System lib: `hidapi` (version 0.15.0) at `/opt/homebrew/Cellar/hidapi/0.15.0/lib`
- Environment: `DYLD_LIBRARY_PATH` configured in user's `.zshrc`
- Device tested: SpaceMouse Compact (successfully detected and read)
- Use case: 6-DOF teleoperation for bimanual manipulation demos

**ManiSkill3 Setup:**
- Repo location: `/Users/larsankile/code/maniskill` (separate workspace)
- Added to Claude's allowed folders for context access
- Installation: `pip install --upgrade mani_skill`
- Requires: PyTorch, Vulkan for rendering
- GPU parallelization: 30,000+ FPS data collection capability
- Bimanual tasks available: `TwoRobotPickCube-v1`, `TwoRobotStackCube-v1`
- Single-arm tasks: `PegInsertionSide-v1`, `PickCube-v1`, `StackCube-v1`, etc.

**Teleoperation Script (`spacemouse_teleop.py`):**
- Control mode: `pd_ee_target_delta_pose` (default, 6-DOF) - prevents drooping
  - Alternative modes: `pd_ee_delta_pose`, `pd_ee_target_delta_pos`, `pd_ee_delta_pos`
- SpaceMouse → Robot mapping:
  - XYZ translation → End-effector position delta (axis mapping: `-yxz`)
  - Roll/Pitch/Yaw rotation → End-effector orientation delta (axis mapping: `-x-yz`)
  - Left button → Toggle gripper
  - Ctrl+C → Quit
- PD controller gains (tuned for teleoperation):
  - Stiffness: 2000 (default 1000)
  - Damping: 200 (default 100)
  - Force limit: 200 (default 100)
- Deadzone: 0.02 (filters sensor noise)
- Adjustable speeds: `--speed` (default 0.15), `--rot-speed` (default 0.3)
- **Recording removed**: Will be re-implemented with LeRobot integration

## What's Next

### IMMEDIATE: Data Pipeline Setup (Session 5+)

**Status (Session 4)**:
- ✅ ManiSkill environment working (installed `pin` for pinocchio)
- ✅ `spacemouse_teleop.py` fully working and tested
- ✅ Teleoperation tested successfully on PegInsertionSide-v1
- ✅ Control issues resolved (responsiveness, frame alignment, drooping)

**Next steps**:
1. **Re-implement trajectory recording in teleoperation script**
   - Add HDF5 recording back (ManiSkill RecordEpisode wrapper or custom)
   - Right button: Save trajectory & reset
   - Ctrl+C: Save all and quit
   - Output location: `demos/<env-id>/spacemouse/`
2. **Collect demo trajectories**
   - Gather 5-10 demos on simple task (PegInsertionSide-v1)
   - Validate recording format
3. **Create ManiSkill → LeRobot data converter**
   - Parse ManiSkill HDF5 trajectory files
   - Convert to LeRobot dataset format
   - Handle: observations, actions, rewards, episode boundaries
   - Add camera images if available
4. Verify LeRobot can load converted data
5. Test simple BC baseline (ACT or Diffusion Policy) with LeRobot
6. Test on bimanual task (`TwoRobotStackCube-v1`) - may need script modifications

**Key technical requirement**:
- Data must be converted from ManiSkill format → LeRobot format
- LeRobot will be used for ALL training (BC, RL, VLA fine-tuning)
- ManiSkill is only for simulation environments

### Infrastructure & Tooling Decisions

**Training Framework**: HuggingFace LeRobot (NOT ManiSkill built-in baselines)
- Rationale: Environment-agnostic, better maintained, state-of-the-art implementations
- Use LeRobot for: Datasets, policies (ACT, Diffusion, VLA), training loops
- Use ManiSkill for: Environments, physics simulation, rendering
- Data pipeline: ManiSkill teleoperation → HDF5 → Convert to LeRobot format
- Need to implement: ManiSkill-to-LeRobot data converter

### Immediate Priorities (De-risking Experiments)

1. **Data infrastructure setup**
   - Create ManiSkill → LeRobot dataset converter
   - Verify LeRobot can load and train on ManiSkill data
   - Test with simple BC baseline (ACT or Diffusion Policy)

2. **Scaling crossover analysis**
   - Simple BC + RL pipeline (using LeRobot)
   - Pre-train with varying data amounts
   - Track online trajectories (not samples) to target performance
   - Goal: Characterize when RL becomes beneficial

3. **Unifying objectives**
   - Simple insertion task
   - Compare: BC-only vs. BC+corrections vs. BC+rollouts+corrections vs. BC+RL+rollouts+corrections
   - Goal: Understand value of different data types

4. **VLA + RL integration**
   - Fine-tune Pi0 on simple task (using LeRobot)
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

- **Robot/simulation environment** ✅ DONE (ManiSkill3)
  - Bimanual arms with parallel-jaw grippers ✅
  - RGB camera(s) for vision ✅
  - Using ManiSkill3 simulation
- **Data collection system** ✅ DONE (SpaceMouse teleoperation)
  - SpaceMouse integration complete
  - Teleoperation script ready for testing
- **Data pipeline** ⚠️ TODO
  - ManiSkill → LeRobot converter (needs implementation)
  - HuggingFace dataset creation/upload
- **BC baseline** ⚠️ TODO (use LeRobot)
  - ACT or Diffusion Policy from LeRobot
  - Action chunking support (built into LeRobot policies)
- **RL fine-tuning pipeline** ⚠️ TODO
  - SAC, PPO, residual methods
  - May need custom integration with LeRobot
  - Need to check what RL support LeRobot has
- **Evaluation metrics** and logging ⚠️ TODO

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
10. **Target delta control**: Use `pd_ee_target_delta_pose` (not regular delta) to prevent drooping
    - Investigated ManiSkill controller architecture (panda.py, pd_ee_pose.py)
    - Regular delta: actions relative to current actual pose → drift accumulates
    - Target delta: actions relative to last commanded pose → no drift accumulation
    - Critical for stable teleoperation when holding objects under load
11. **PD controller gains**: Doubled from defaults (stiffness 2000, damping 200, force_limit 200)
    - More responsive to SpaceMouse inputs
    - Stronger position holding against gravity and external forces
12. **Frame alignment**: Implemented axis remapping system (`-yxz` translation, `-x-yz` rotation)
    - Makes teleoperation intuitive (forward=forward, left=left)
    - Allows easy adjustment without code changes

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

- Trajectory recording removed from teleoperation (needs re-implementation)
- No ManiSkill → LeRobot data converter yet
- No BC/RL pipeline implemented yet
- Teleoperation only tested on single-arm tasks (bimanual needs testing)

## Tips for Future Claude Sessions

1. **Research context**: This is about continual learning and RL fine-tuning, NOT just generic "self-improving robots"
2. **SpaceMouse**: Teleoperation tool for data collection - already set up and working
3. **Focus**: Bimanual, long-horizon, demo-driven manipulation tasks
4. **Methods**: Unifying BC + DAgger + RL with action chunking and VLAs
5. **User preferences**: Clear reproducible setup, automation scripts, thorough documentation
6. **Control mode importance**: Always use `pd_ee_target_delta_pose` for teleoperation (not `pd_ee_delta_pose`)
   - Target-based control prevents drooping/drift when holding objects
   - Regular delta control accumulates error and drifts over time
   - This is critical for stable teleoperation with gripper holding objects

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
- ManiSkill3 docs: https://maniskill.readthedocs.io/
- ManiSkill3 repo: `/Users/larsankile/code/maniskill` (in Claude's allowed folders)
- Research presentation: `/Users/larsankile/Downloads/Project-dicussion.pdf`

## Files Created This Session (Session 2)

**New Python scripts:**
- `spacemouse_teleop.py` - Main SpaceMouse teleoperation for ManiSkill
- `test_maniskill_env.py` - Quick test script for ManiSkill environments

**New documentation:**
- `SPACEMOUSE_TELEOP.md` - Comprehensive teleoperation guide
  - Usage instructions
  - Available tasks
  - Control modes
  - Speed adjustment
  - Output data format
  - Troubleshooting

**Updated files:**
- `README.md` - Added ManiSkill setup and teleoperation sections
- `.claude/notes.md` - This file, updated with Session 2 progress

---

*Last updated: 2025-10-13 Session 4*
*Status: SpaceMouse teleoperation fully working and tested. Ready for data collection pipeline.*
