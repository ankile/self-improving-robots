"""
Test script for the Robosuite SpaceMouse device.

This tests the minimal SpaceMouse driver without requiring Robosuite.
"""

import time

from sir.teleoperation.robosuite_spacemouse import RobosuiteSpaceMouse


def main():
    """Test the Robosuite SpaceMouse device."""
    print("Testing Robosuite SpaceMouse device")
    print("=" * 60)

    try:
        # Initialize device
        device = RobosuiteSpaceMouse()
        device.start_control()

        print("\nDevice initialized successfully!")
        print("Move the SpaceMouse and press buttons to test.")
        print("Press Ctrl+C to stop.\n")

        # Test loop
        last_gripper = device.control_gripper
        try:
            while True:
                control = device.control
                gripper = device.control_gripper

                # Print control values
                print(
                    f"Pos: [{control[0]:6.3f}, {control[1]:6.3f}, {control[2]:6.3f}]  "
                    f"Rot: [{control[3]:6.3f}, {control[4]:6.3f}, {control[5]:6.3f}]  "
                    f"Gripper: {gripper}  ",
                    end="\r",
                )

                # Detect gripper state changes
                if gripper != last_gripper:
                    state = "CLOSED" if gripper == 1 else "OPEN"
                    print(f"\nGripper state changed: {state}")
                last_gripper = gripper

                # Check for reset
                if device.reset_requested:
                    print("\nReset button pressed!")
                    break

                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. SpaceMouse is connected")
        print("  2. No other process is using the device (killall 3DconnexionHelper)")
        print("  3. hidapi is installed (pip install hidapi)")
        return 1

    finally:
        if "device" in locals():
            device.close()
        print("\nTest complete!")

    return 0


if __name__ == "__main__":
    exit(main())
