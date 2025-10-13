# SpaceMouse Teleoperation Guide

This guide explains how to use the SpaceMouse to collect demonstrations in ManiSkill environments.

## Overview

The `spacemouse_teleop.py` script provides a simple interface for teleoperation:
- **6-DOF control**: SpaceMouse provides intuitive 3D position + 3D rotation control
- **Automatic recording**: Demonstrations are saved in ManiSkill's HDF5 format
- **Button controls**: Toggle gripper and save trajectories with SpaceMouse buttons
- **Any ManiSkill task**: Works with all single-arm ManiSkill environments

## Quick Start

1. **Test your setup**:
   ```bash
   # Test SpaceMouse connection
   python test_spacemouse.py

   # Test ManiSkill environment
   python test_maniskill_env.py
   ```

2. **Start teleoperation**:
   ```bash
   python spacemouse_teleop.py -e PegInsertionSide-v1
   ```

3. **Control the robot**:
   - Move SpaceMouse: Robot end-effector follows your movements
   - Left button: Toggle gripper (press to open/close)
   - Right button: Save current trajectory and reset for new one
   - Ctrl+C in terminal: Quit and save all data

## Available Tasks

### Single-Arm Manipulation Tasks

**Simple tasks** (good for testing):
- `PickCube-v1` - Pick and place a cube
- `PushCube-v1` - Push a cube to a target
- `StackCube-v1` - Stack two cubes
- `LiftPegUpright-v1` - Lift a peg upright

**Precision tasks** (more challenging):
- `PegInsertionSide-v1` - Insert peg into hole (recommended starting point)
- `PlugCharger-v1` - Plug a charger into receptacle
- `PushT-v1` - Push T-shaped block to target

**Long-horizon tasks**:
- `StackPyramid-v1` - Stack 3 cubes in pyramid
- `AssemblingKits-v1` - Insert shape into kit

For a complete list, see [ManiSkill docs](https://maniskill.readthedocs.io/en/latest/tasks/index.html).

## Control Modes

The script supports different control modes via `--control-mode`:

### `pd_ee_delta_pose` (default)
- **Action space**: 7D (3 translation + 3 rotation + 1 gripper)
- **Best for**: Precision tasks requiring orientation control
- **SpaceMouse mapping**:
  - XYZ movement → End-effector position
  - Roll/Pitch/Yaw → End-effector orientation
  - Left button → Gripper toggle

### `pd_ee_delta_pos`
- **Action space**: 4D (3 translation + 1 gripper)
- **Best for**: Tasks where orientation doesn't matter
- **SpaceMouse mapping**:
  - XYZ movement → End-effector position
  - Rotation ignored
  - Left button → Gripper toggle

## Speed Adjustment

Control how responsive the robot is to SpaceMouse movements:

```bash
# Slower, more precise control (default)
python spacemouse_teleop.py -e PegInsertionSide-v1 --speed 0.05 --rot-speed 0.1

# Faster control for reaching/transport
python spacemouse_teleop.py -e PegInsertionSide-v1 --speed 0.15 --rot-speed 0.3

# Very slow for fine manipulation
python spacemouse_teleop.py -e PegInsertionSide-v1 --speed 0.02 --rot-speed 0.05
```

**Tips**:
- Start with default speeds and adjust based on feel
- Lower speeds give more precision but slower movement
- Higher speeds allow faster demonstrations but less control

## Output Data Format

Demonstrations are saved in ManiSkill's standard format:

```
demos/
└── PegInsertionSide-v1/
    └── spacemouse/
        ├── trajectory.h5          # All trajectories
        └── trajectory.json        # Metadata
```

**HDF5 structure**:
- `traj_0/`, `traj_1/`, ... - Individual trajectories
  - `actions`: Robot actions taken
  - `obs`: Observations (if obs_mode != "none")
  - `env_states`: Full simulator states
  - `success`: Whether task succeeded

**Compatible with**:
- ManiSkill BC baselines
- ManiSkill Diffusion Policy baseline
- Custom training scripts

## Tips for Good Demonstrations

1. **Practice first**: Do a few test runs without saving (Ctrl+C to abort)
2. **Smooth motions**: SpaceMouse allows smooth, natural movements
3. **Break into segments**: Use right button to save and start new trajectory
4. **Collect failures too**: Both successes and failures can be useful for research
5. **Vary initial conditions**: ManiSkill randomizes objects on reset

## Troubleshooting

### SpaceMouse not detected
- Check USB connection
- Verify permissions: System Settings → Privacy & Security → Input Monitoring
- Kill 3Dconnexion driver: `killall 3DconnexionHelper`
- Test with: `python test_spacemouse.py`

### Environment won't open
- Check ManiSkill installation: `pip show mani-skill`
- Check Vulkan setup (needed for rendering)
- Test with: `python test_maniskill_env.py`

### Robot moves too fast/slow
- Adjust `--speed` and `--rot-speed` parameters
- Default: `--speed 0.05 --rot-speed 0.1`

### Actions feel "sluggish"
- Try `pd_ee_delta_pos` instead of `pd_ee_delta_pose`
- Increase `--speed` parameter
- Lower `--max-episode-steps` if environment times out

### Gripper doesn't work
- Ensure you're using a robot with gripper (default: `panda`)
- Check that left button is being detected: watch terminal output

## Advanced Usage

### Custom robot
```bash
python spacemouse_teleop.py -e PickCube-v1 --robot-uid panda_wristcam
```

### Custom episode length
```bash
python spacemouse_teleop.py -e PickCube-v1 --max-episode-steps 200
```

### Full command line options
```bash
python spacemouse_teleop.py --help
```

## Next Steps

After collecting demonstrations:
1. **Visualize**: Use ManiSkill's replay tools to verify quality
2. **Train BC policy**: Use ManiSkill's BC baseline
3. **Train Diffusion Policy**: Use ManiSkill's diffusion baseline
4. **Fine-tune with RL**: Use demonstrations to initialize RL policy

See ManiSkill documentation for training baselines:
- [Learning from Demonstrations](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html)
- [Reinforcement Learning](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/index.html)

## Technical Details

### Control Frame
- Translation: "root frame" (world-aligned, positioned at robot base)
- Rotation: "root-aligned body rotation" (world-aligned, positioned at end-effector)
- This provides intuitive camera-relative control

### Coordinate System
- X: forward/backward (red axis)
- Y: left/right (green axis)
- Z: up/down (blue axis)
- Roll: rotation around X
- Pitch: rotation around Y
- Yaw: rotation around Z

### Action Normalization
Actions are NOT normalized by default. The `--speed` parameters directly scale the SpaceMouse readings (which are in range [-1, 1]) to appropriate delta values for the robot.

## Related Files

- `spacemouse_teleop.py` - Main teleoperation script
- `test_spacemouse.py` - Test SpaceMouse connection
- `test_maniskill_env.py` - Test ManiSkill environment
- `setup_spacemouse.sh` - SpaceMouse setup automation
- `SETUP_SPACEMOUSE.md` - Detailed SpaceMouse setup guide
