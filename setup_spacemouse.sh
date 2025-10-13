#!/bin/bash
# Automated SpaceMouse setup script for macOS
# This script installs all required dependencies for SpaceMouse support

set -e  # Exit on error

echo "🎮 SpaceMouse Setup Script for macOS"
echo "===================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "✓ Found Python: $($PYTHON_CMD --version)"
echo ""

# Install hidapi via Homebrew
echo "📦 Installing hidapi via Homebrew..."
if brew list hidapi &> /dev/null; then
    echo "✓ hidapi is already installed"
else
    brew install hidapi
    echo "✓ hidapi installed successfully"
fi
echo ""

# Get hidapi installation path
HIDAPI_PATH=$(brew --prefix hidapi)
echo "✓ hidapi path: $HIDAPI_PATH"
echo ""

# Install Python packages
echo "🐍 Installing Python packages..."
$PYTHON_CMD -m pip install pyspacemouse
echo "✓ pyspacemouse installed"
echo ""

# Check if ARM Mac
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "🔧 Detected ARM Mac (Apple Silicon)"
    echo "   Installing patched easyhid for ARM support..."
    $PYTHON_CMD -m pip uninstall -y easyhid 2>/dev/null || true
    $PYTHON_CMD -m pip install git+https://github.com/bglopez/python-easyhid.git
    echo "✓ Patched easyhid installed"
else
    echo "ℹ️  Intel Mac detected - using standard easyhid"
fi
echo ""

# Configure environment variable
echo "⚙️  Configuring environment variables..."

# Detect shell
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ] || [ "$SHELL" = "/bin/bash" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.zshrc"  # Default to zsh
fi

EXPORT_LINE="export DYLD_LIBRARY_PATH=${HIDAPI_PATH}/lib:\$DYLD_LIBRARY_PATH"

# Check if already configured
if grep -q "DYLD_LIBRARY_PATH.*hidapi" "$SHELL_RC" 2>/dev/null; then
    echo "✓ Environment variable already configured in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "# SpaceMouse support - hidapi library path" >> "$SHELL_RC"
    echo "$EXPORT_LINE" >> "$SHELL_RC"
    echo "✓ Added environment variable to $SHELL_RC"
fi
echo ""

# Apply environment variable to current session
export DYLD_LIBRARY_PATH=${HIDAPI_PATH}/lib:$DYLD_LIBRARY_PATH

echo "✅ SpaceMouse setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Reload your shell: source $SHELL_RC"
echo "   2. Or open a new terminal window"
echo "   3. Test your setup: python test_spacemouse.py"
echo ""
echo "🔍 Troubleshooting:"
echo "   - Make sure your SpaceMouse is plugged in"
echo "   - Grant Input Monitoring permissions (System Settings → Privacy & Security)"
echo "   - If 3Dconnexion drivers are installed: killall 3DconnexionHelper"
echo ""
