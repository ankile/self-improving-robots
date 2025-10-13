#!/usr/bin/env python3
"""
Test script for reading SpaceMouse values on macOS
"""
import pyspacemouse
import time

def main():
    print("Attempting to connect to SpaceMouse...")

    # Open connection to SpaceMouse
    success = pyspacemouse.open(
        dof_callback=None,  # We'll print manually for more control
        button_callback=None
    )

    if success:
        print("✓ SpaceMouse connected successfully!")
        print("\nReading SpaceMouse values... (Press Ctrl+C to stop)\n")

        try:
            while True:
                state = pyspacemouse.read()

                # Print the state values
                print(f"\rPosition: x={state.x:7.2f}, y={state.y:7.2f}, z={state.z:7.2f} | "
                      f"Rotation: roll={state.roll:7.2f}, pitch={state.pitch:7.2f}, yaw={state.yaw:7.2f} | "
                      f"Buttons: {state.buttons}", end='', flush=True)

                time.sleep(0.01)  # 100Hz update rate

        except KeyboardInterrupt:
            print("\n\nStopping SpaceMouse reader...")
            pyspacemouse.close()
            print("Connection closed.")
    else:
        print("✗ Failed to connect to SpaceMouse")
        print("\nTroubleshooting tips:")
        print("1. Make sure your SpaceMouse is plugged in")
        print("2. If you have 3Dconnexion drivers installed, quit '3DconnexionHelper':")
        print("   killall 3DconnexionHelper")
        print("3. Check that you have proper permissions (System Settings → Privacy & Security → Input Monitoring)")

if __name__ == "__main__":
    main()
