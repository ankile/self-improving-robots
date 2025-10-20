# SpaceMouse Setup Guide

This guide documents how to set up SpaceMouse support on macOS (tested on ARM/Apple Silicon Macs).

## Prerequisites

- macOS (tested on Apple Silicon M1/M2/M3)
- Homebrew package manager
- Python 3.11+ (via micromamba or other Python distribution)

## Installation Steps

### 1. Install System Dependencies

Install `hidapi` via Homebrew:
```bash
brew install hidapi
```

### 2. Install Python Packages

Install the main SpaceMouse library:
```bash
pip install pyspacemouse
```

For ARM Macs (M1/M2/M3), install the patched version of easyhid:
```bash
pip uninstall -y easyhid
pip install git+https://github.com/bglopez/python-easyhid.git
```

### 3. Configure Environment Variables

Add the hidapi library path to your shell configuration:

For **zsh** (default on macOS):
```bash
echo 'export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hidapi/0.15.0/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

For **bash**:
```bash
echo 'export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hidapi/0.15.0/lib:$DYLD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Note**: Update the version number (0.15.0) if you have a different version of hidapi. Check with:
```bash
brew --prefix hidapi
```

## Automated Setup

Alternatively, run the automated setup script:
```bash
./setup_spacemouse.sh
```

## Testing

Run the test script to verify your SpaceMouse is working:
```bash
python -m sir.tests.test_spacemouse
```

You should see:
- "SpaceMouse Compact found" (or your specific model)
- "✓ SpaceMouse connected successfully!"
- Real-time position, rotation, and button values

Press Ctrl+C to stop.

## Troubleshooting

### Device not found
1. Make sure your SpaceMouse is plugged in
2. Check USB connection and try a different port

### Permission issues
Grant Input Monitoring permissions:
- System Settings → Privacy & Security → Input Monitoring
- Add your terminal application or IDE

### Conflicts with 3Dconnexion drivers
If you have official 3Dconnexion drivers installed, quit the helper process:
```bash
killall 3DconnexionHelper
```

### Library loading errors
Verify hidapi is installed and the path is correct:
```bash
ls -la $(brew --prefix hidapi)/lib/
```

## Supported Devices

- SpaceNavigator
- SpaceMouse Pro
- SpaceMouse Pro Wireless
- SpaceMouse Wireless
- SpaceMouse Compact
- SpacePilot
- And other 3Dconnexion devices

## Resources

- [PySpaceMouse Documentation](https://spacemouse.kubaandrysek.cz/)
- [PySpaceMouse GitHub](https://github.com/JakubAndrysek/PySpaceMouse)
- [hidapi Homepage](https://github.com/libusb/hidapi)
