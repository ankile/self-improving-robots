# Robosuite Teleoperation

This directory contains minimal teleoperation code for Robosuite environments, adapted from the official Robosuite implementation.

## Overview

The code here allows you to:
- Teleoperate Robosuite environments with a SpaceMouse
- Collect demonstration data for conversion to LeRobot format
- Use the same SpaceMouse device for both ManiSkill and Robosuite

## Key Differences from Robosuite

This implementation is **minimal** and focused on teleoperation only:

- Uses `hidapi` directly (same as Robosuite) for low-level device access
- Does NOT include Robosuite's HDF5 data collection format
- Does NOT include the full Device base class interface
- Simpler, focused solely on collecting raw episode data for LeRobot

## Files

- `robosuite_spacemouse.py` - Minimal SpaceMouse device driver
- `robosuite_teleop.py` - Main teleoperation script
- `ROBOSUITE_README.md` - This file

## Installation

1. Install the base package in editable mode:
```bash
pip install -e .
```

2. Install hidapi system dependency (macOS):
```bash
brew install hidapi
```

3. Install Robosuite (optional dependency):
```bash
pip install -e ".[robosuite]"
# OR directly: pip install robosuite
```

## Usage

### Basic Teleoperation

**IMPORTANT for macOS**: You must use `mjpython` instead of `python` to run the viewer on macOS:

```bash
# macOS (REQUIRED - use mjpython)
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda

# Linux (use python)
python -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda
```

The MuJoCo viewer requires GUI operations on the main thread, which is only supported by `mjpython` on macOS. Using regular `python` will crash with an NSWindow error.

### Different Environments

```bash
# Pick and place
mjpython -m sir.teleoperation.robosuite_teleop --env PickPlaceCan --robot Sawyer

# Two-arm bimanual
mjpython -m sir.teleoperation.robosuite_teleop --env TwoArmLift --robot Baxter --config bimanual

# Two-arm with two robots
mjpython -m sir.teleoperation.robosuite_teleop --env TwoArmLift --robot Panda --config parallel
```

### Camera Observations and Image Saving

```bash
# Enable camera observations and save example images
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
  --save-images \
  --cameras "agentview,robot0_eye_in_hand" \
  --camera-height 256 \
  --camera-width 256
```

### Adjust Sensitivity

```bash
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
  --pos-sensitivity 2.0 \
  --rot-sensitivity 1.5
```

### Test Device Only

Test the SpaceMouse without running Robosuite:

```bash
python -m sir.tests.test_robosuite_spacemouse
```

## SpaceMouse Controls

- **Move laterally** - Move arm in x-y plane
- **Move vertically** - Move arm up/down
- **Twist/rotate** - Rotate end-effector
- **Left button** - Toggle gripper open/close
- **Right button** - Reset episode
- **Ctrl+C** - Quit

## Technical Notes

### macOS Requirement: mjpython

**CRITICAL**: On macOS, you MUST use `mjpython` instead of `python` to run Robosuite teleoperation with the viewer.

Why? The MuJoCo viewer creates GUI windows using NSWindow, which on macOS must be instantiated on the main thread. Regular Python runs GUI code on background threads, causing crashes. MuJoCo's `mjpython` wrapper ensures GUI operations run on the main thread.

Error you'll see if using regular `python`:
```
*** Terminating app due to uncaught exception 'NSInternalInconsistencyException',
reason: 'NSWindow should only be instantiated on the main thread!'
```

**Solution**: Always use `mjpython` on macOS:
```bash
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda
```

The `mjpython` executable is installed automatically when you install MuJoCo via pip.

### Device Access

The code uses `hid` library for direct USB HID access:
- Vendor ID: `0x256F` (3Dconnexion)
- Product ID: `0xC635` (SpaceMouse Compact, default)

**Note**: Different SpaceMouse models may have different product IDs. To detect your device IDs, run:
```bash
python -c "import hid; [print(f'Vendor: 0x{d[\"vendor_id\"]:04X}, Product: 0x{d[\"product_id\"]:04X}, Name: {d[\"product_string\"]}') for d in hid.enumerate() if 'space' in d['product_string'].lower()]"
```

### Conflicts

If the SpaceMouse isn't detected, other processes may be using it:
```bash
killall 3DconnexionHelper
```

### Coordinate Frames

The SpaceMouse inputs are mapped to robot actions as:
- Translation: y→x, x→y, -z→z (hardware frame → robot frame)
- Rotation: Roll, pitch, yaw with appropriate sign flips
- Gripper: Binary toggle (0=open, 1=closed) → continuous (-1 to 1)

## Future Work

- [ ] Add LeRobot dataset conversion
- [ ] Support camera observations
- [ ] Add episode metadata tracking
- [ ] Support multi-arm control switching
- [ ] Add keyboard controls as fallback

## Credits

Adapted from [Robosuite](https://github.com/ARISE-Initiative/robosuite) by ARISE Initiative.
Original SpaceMouse implementation by the Robosuite team.
