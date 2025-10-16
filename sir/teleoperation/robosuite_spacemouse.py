"""
Minimal SpaceMouse device driver for Robosuite teleoperation.

Adapted from robosuite's SpaceMouse implementation to work with our codebase
for collecting demonstrations in LeRobot format.

This uses the `hid` library (not pyspacemouse) to directly interface with the device.
"""

import threading
import time
from collections import namedtuple

import numpy as np

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "Install with: pip install hidapi"
    ) from exc

# SpaceMouse vendor/product IDs
SPACEMOUSE_VENDOR_ID = 0x256F  # 3Dconnexion
SPACEMOUSE_PRODUCT_ID = 0xC635  # SpaceMouse Compact (default)

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class RobosuiteSpaceMouse:
    """
    Minimal SpaceMouse driver for Robosuite teleoperation.

    This class provides direct HID access to the SpaceMouse without
    requiring the full Robosuite Device interface.

    Args:
        vendor_id (int): USB vendor ID
        product_id (int): USB product ID
        pos_sensitivity (float): Position control sensitivity multiplier
        rot_sensitivity (float): Rotation control sensitivity multiplier
    """

    def __init__(
        self,
        vendor_id=SPACEMOUSE_VENDOR_ID,
        product_id=SPACEMOUSE_PRODUCT_ID,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()

        try:
            self.device.open(self.vendor_id, self.product_id)
        except OSError as e:
            print(
                "Failed to open SpaceMouse device. "
                "Consider killing other processes that may be using the device:\n"
                "  killall 3DconnexionHelper"
            )
            raise

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        print(f"Manufacturer: {self.device.get_manufacturer_string()}")
        print(f"Product: {self.device.get_product_string()}")

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        # Button states
        self.gripper_closed = False
        self._reset_requested = False

        # Control state
        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._enabled = False

        # Launch listener thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """Print control instructions."""
        print("\nSpaceMouse Controls:")
        print("  Left button (click)     - toggle gripper open/close")
        print("  Right button            - reset episode")
        print("  Move mouse laterally    - move arm horizontally in x-y plane")
        print("  Move mouse vertically   - move arm vertically")
        print("  Twist mouse about axis  - rotate arm about corresponding axis")
        print("  Control+C               - quit")
        print()

    def start_control(self):
        """Enable control."""
        self._enabled = True
        self._reset_requested = False

    def _run(self):
        """Listener thread that continuously reads from the device."""
        t_last_click = -1

        while True:
            try:
                d = self.device.read(13)
            except OSError:
                # Device was closed or disconnected
                break

            if d is not None and self._enabled:
                if self.product_id == 50741:
                    # Older SpaceMouse model - separate messages for pos/rot
                    if d[0] == 1:  # Position data
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                    elif d[0] == 2:  # Rotation data
                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                else:
                    # Modern SpaceMouse - all 6-DOF in one message
                    if d[0] == 1:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

                # Button handling
                if d[0] == 3:
                    # Left button - toggle gripper
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        # Debounce
                        if elapsed_time > 0.3 or elapsed_time < 0:
                            self.gripper_closed = not self.gripper_closed

                    # Right button - reset
                    if d[1] == 2:
                        self._reset_requested = True
                        self._enabled = False

    @property
    def control(self):
        """
        Get current 6-DOF control values.

        Returns:
            np.array: [x, y, z, roll, pitch, yaw]
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Get gripper state.

        Returns:
            int: 1 for closed, 0 for open
        """
        return 1 if self.gripper_closed else 0

    @property
    def reset_requested(self):
        """Check if reset was requested."""
        return self._reset_requested

    def close(self):
        """Close the device connection."""
        self._enabled = False
        self.device.close()


if __name__ == "__main__":
    # Simple test
    space_mouse = RobosuiteSpaceMouse()
    space_mouse.start_control()

    print("Testing SpaceMouse. Move the device and press buttons. Ctrl+C to stop.\n")

    try:
        while True:
            control = space_mouse.control
            gripper = space_mouse.control_gripper
            print(f"Control: {control}  Gripper: {gripper}", end="\r")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        space_mouse.close()
