"""Utility functions for teleoperation."""

import numpy as np
from pynput import keyboard


def apply_deadzone(value: float, threshold: float) -> float:
    """
    Apply deadzone to SpaceMouse input to filter out noise.

    Args:
        value: Input value from SpaceMouse
        threshold: Deadzone threshold (values below this are set to 0)

    Returns:
        Filtered value (0.0 if below threshold, otherwise original value)
    """
    return value if abs(value) > threshold else 0.0


def parse_axis_mapping(mapping_str: str):
    """
    Parse axis mapping string into remapping instructions.

    Examples:
        "xyz" -> no change
        "yxz" -> swap x and y
        "-xyz" -> negate x
        "-x-yz" -> negate x and y

    Args:
        mapping_str: String describing the axis mapping (e.g., "-yxz")

    Returns:
        A function that takes (x, y, z) and returns remapped (x', y', z')

    Raises:
        ValueError: If mapping string is invalid
    """
    mapping_str = mapping_str.lower().strip()

    # Build axis lookup
    axis_map = {}
    axis_chars = [c for c in mapping_str if c in 'xyz']

    if len(axis_chars) != 3 or len(set(axis_chars)) != 3:
        raise ValueError(
            f"Invalid axis mapping: {mapping_str}. Must contain x, y, z exactly once."
        )

    # Parse each axis and its sign
    for i, char in enumerate(['x', 'y', 'z']):
        # Find where this output axis comes from
        idx = axis_chars.index(char)
        # Check if there's a negative sign before it
        sign_idx = mapping_str.index(char)
        is_negative = sign_idx > 0 and mapping_str[sign_idx - 1] == '-'
        axis_map[i] = (idx, -1.0 if is_negative else 1.0)

    def remap(x, y, z):
        vals = [x, y, z]
        return tuple(vals[axis_map[i][0]] * axis_map[i][1] for i in range(3))

    return remap


class KeyboardListener:
    """System-wide keyboard listener that works regardless of window focus."""

    def __init__(self):
        self.last_key = None
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        """Callback for key press events."""
        try:
            # Handle alphanumeric keys
            if hasattr(key, 'char') and key.char:
                self.last_key = key.char
        except AttributeError:
            # Handle special keys (if needed in the future)
            pass

    def read_key(self):
        """Read the last pressed key and clear it."""
        key = self.last_key
        self.last_key = None
        return key

    def close(self):
        """Stop the keyboard listener."""
        self.listener.stop()
