# Robosuite macOS CGL Backend Fix

**Date**: October 17, 2025
**MuJoCo Version**: 3.3.7
**Issue**: Cannot use onscreen viewer + camera observations simultaneously on macOS
**Status**: ✅ FIXED

## Problem

When attempting to run Robosuite teleoperation with both onscreen viewer (`mjviewer`) and camera observations on macOS, the following error occurred:

```
*** Terminating app due to uncaught exception 'NSInternalInconsistencyException',
reason: 'NSWindow should only be instantiated on the main thread!'
```

### Root Cause

1. **MuJoCo's mjpython** runs Python code on a background thread (keeps main thread free for viewer)
2. **Offscreen rendering** (for camera observations) tried to use GLFW backend
3. **GLFW on macOS** requires main thread for OpenGL context creation
4. **Conflict**: Background thread trying to create GLFW context → NSWindow crash

### Why This Happened

Robosuite had **incomplete CGL support**:
- Code validated `MUJOCO_GL=cgl` as acceptable for macOS (line 67 of `binding_utils.py`)
- **BUT** no code path actually used CGL - always fell through to GLFW (line 75)
- Result: MuJoCo 2.3.4+'s thread-safe CGL backend was never utilized

## Solution

### What is CGL?

**CGL (Core OpenGL)** is Apple's low-level OpenGL API:
- Creates contexts **not tied to Cocoa windows**
- Can create contexts on **any thread** (not just main)
- Available in MuJoCo 2.3.4+ as `mujoco.cgl.GLContext`
- Specifically designed to solve macOS threading issues

### Implementation

#### 1. Created CGL Context Wrapper

**File**: `/Users/larsankile/code/robosuite/robosuite/renderers/context/cgl_context.py`

```python
from mujoco.cgl import GLContext

class CGLGLContext(GLContext):
    """An OpenGL context created via CGL for macOS.

    CGL (Core OpenGL) creates contexts that are not tied to Cocoa windows,
    allowing offscreen rendering on background threads on macOS.
    This solves the main thread requirement that GLFW has on macOS.
    """

    def __init__(self, max_width, max_height, device_id=0):
        # CGL doesn't use device_id, but we accept it for API compatibility
        super().__init__(max_width, max_height)
```

#### 2. Updated Robosuite Binding Utils

**File**: `/Users/larsankile/code/robosuite/robosuite/utils/binding_utils.py`

Added CGL case to context selection logic (line 74-75):

```python
elif _SYSTEM == "Darwin" and _MUJOCO_GL == "cgl":
    from robosuite.renderers.context.cgl_context import CGLGLContext as GLContext
```

#### 3. Updated Teleoperation Script

**File**: `/Users/larsankile/code/self-improving-robots/sir/teleoperation/robosuite_teleop.py`

Added automatic CGL backend selection for macOS:

```python
# On macOS, force CGL backend for offscreen rendering when using camera observations
# CGL doesn't require main thread, unlike GLFW
if platform.system() == "Darwin":
    if "MUJOCO_GL" not in os.environ or os.environ.get("MUJOCO_GL") == "":
        os.environ["MUJOCO_GL"] = "cgl"
```

## Results

### Before Fix
- ❌ Onscreen viewer + camera obs → CRASH
- ✅ Headless mode (no viewer) + camera obs → Works
- ✅ Onscreen viewer only (no camera obs) → Works

### After Fix
- ✅ Onscreen viewer + camera obs → **WORKS!**
- ✅ Headless mode + camera obs → Works
- ✅ Onscreen viewer only → Works

### Working Command

```bash
mjpython -m sir.teleoperation.robosuite_teleop --env Lift --robot Panda \
  --save-images \
  --cameras "agentview,robot0_eye_in_hand"
```

**Features**:
- ✅ Onscreen mjviewer window (visual feedback)
- ✅ Camera observations in observation dict
- ✅ Example images saved to `./robosuite_images/`
- ✅ No threading conflicts
- ✅ Full teleoperation capability

## Technical Details

### Why CGL Works

1. **GLFW Limitation**:
   - Uses Cocoa's NSOpenGL API
   - Contexts tied to NSWindow objects
   - NSWindow must be created on main thread

2. **CGL Advantage**:
   - Low-level Core OpenGL API
   - Contexts independent of windowing system
   - Can be created on any thread
   - No Cocoa dependency

3. **MuJoCo Integration**:
   - MuJoCo 2.3.4 (2023) added CGL support
   - Automatically uses CGL when `MUJOCO_GL=cgl`
   - Designed specifically for macOS headless rendering

### Architecture

```
┌─────────────────────────────────────┐
│  mjpython (Main Thread)             │
│  ├─ NSWindow (Viewer)               │
│  └─ Python code runs on bg thread   │
└─────────────────────────────────────┘
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│  Onscreen        │          │  Offscreen       │
│  Rendering       │          │  Rendering       │
│  (Viewer)        │          │  (Camera Obs)    │
│                  │          │                  │
│  Uses: GLFW      │          │  Uses: CGL       │
│  Thread: Main    │          │  Thread: Any     │
└──────────────────┘          └──────────────────┘
```

## Impact

### For This Project
- Enables proper data collection with visual feedback
- No more blind teleoperation in headless mode
- Better workflow for demonstration collection

### For Robosuite Community
- Fixes longstanding macOS limitation
- Clean, minimal implementation (3 files changed)
- Compatible with existing code
- Could benefit other macOS users

### Potential Contribution
This fix could be contributed back to Robosuite:
- **Files Changed**: 2
- **Files Created**: 1
- **Lines of Code**: ~40 total
- **Backward Compatible**: Yes (falls back to GLFW if needed)

## References

- MuJoCo 2.3.4 Release Notes: https://github.com/google-deepmind/mujoco/discussions/828
- CGL Documentation: https://developer.apple.com/documentation/coregraphics
- MuJoCo Python Bindings: https://mujoco.readthedocs.io/en/stable/python.html
- Issue Discussion: https://github.com/google-deepmind/mujoco/issues/798

## Validation

Tested on:
- **Platform**: macOS (Apple Silicon)
- **MuJoCo**: 3.3.7
- **Python**: 3.11
- **Robosuite**: Latest (cloned from source)
- **Environment**: Lift task with Panda robot
- **Cameras**: agentview, robot0_eye_in_hand
- **Resolution**: 256x256

## Future Work

- Consider submitting PR to Robosuite upstream
- Test on other macOS versions (Intel Macs)
- Document in Robosuite's official documentation
- Check if similar fix needed for other frameworks (robomimic, etc.)
