#!/bin/bash
#
# SLURM Server Setup Script
#
# This script sets up a complete environment for training on a SLURM-based cluster.
#
# Usage: 
#   From inside self-improving-robots: bash scripts/slurm_setup.sh [BASE_DIR] [ENV_NAME]
#   From work folder: bash self-improving-robots/scripts/slurm_setup.sh [BASE_DIR] [ENV_NAME]
#
# The script will:
# 1. Clone/update all required repositories (lerobot, robosuite, maniskill) as siblings
# 2. Create a micromamba/mamba/conda environment with all dependencies
# 3. Install all packages in editable mode
# 4. Verify the installation
#
# Note: All repos (self-improving-robots, lerobot, robosuite, maniskill) will be
#       placed as siblings in the same parent directory (work folder)
#

set -e  # Exit on any error

# Configuration
BASE_DIR="${1:-.}"
ENV_NAME="${2:-sir}"
CONDA_INIT_FILE=""

# Detect if we're running from inside self-improving-robots and adjust BASE_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */self-improving-robots/scripts ]]; then
    # We're inside self-improving-robots, go up to parent work folder
    if [ "$BASE_DIR" = "." ]; then
        BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
        echo "Detected script running from self-improving-robots/"
        echo "Using parent directory as BASE_DIR: $BASE_DIR"
        echo ""
    fi
fi

# Try to initialize micromamba if it exists in common locations but isn't in PATH
if ! command -v micromamba &> /dev/null; then
    # Check common micromamba installation locations
    for MICROMAMBA_PATH in \
        "$HOME/.local/bin/micromamba" \
        "$HOME/micromamba/bin/micromamba" \
        "/opt/micromamba/bin/micromamba" \
        "$MAMBA_ROOT_PREFIX/bin/micromamba"; do
        if [ -x "$MICROMAMBA_PATH" ]; then
            echo "Found micromamba at: $MICROMAMBA_PATH"
            echo "Initializing micromamba..."
            eval "$("$MICROMAMBA_PATH" shell hook --shell bash)"
            break
        fi
    done
fi

# Detect micromamba/mamba/conda command (prefer micromamba)
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: Neither micromamba, mamba, nor conda found in PATH."
    echo ""
    echo "Please try one of the following:"
    echo ""
    echo "1. Load a module (if on SLURM cluster):"
    echo "   module load micromamba"
    echo "   # or: module load miniconda3"
    echo ""
    echo "2. Initialize micromamba if already installed:"
    echo "   eval \"\$(micromamba shell hook --shell bash)\""
    echo ""
    echo "3. Install micromamba (quick install):"
    echo "   \"\${SHELL}\" <(curl -L micro.mamba.pm/install.sh)"
    echo ""
    echo "4. Install from other sources:"
    echo "   micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    echo "   mamba: https://github.com/conda-forge/miniforge"
    echo "   conda: https://www.anaconda.com/download"
    exit 1
fi

echo "============================================================"
echo "SLURM Server Setup Script"
echo "============================================================"
echo "Conda command: $CONDA_CMD"
echo "Environment name: $ENV_NAME"
echo "Base directory: $BASE_DIR"
echo "============================================================"
echo ""

# Step 1: Create base directory if needed
if [ ! -d "$BASE_DIR" ]; then
    echo "Creating base directory: $BASE_DIR"
    mkdir -p "$BASE_DIR"
fi
cd "$BASE_DIR"

# Step 2: Clone/update required repositories
echo "Step 1: Setting up repositories..."
echo "  - self-improving-robots"
if [ ! -d "self-improving-robots" ]; then
    git clone https://github.com/ankile/self-improving-robots.git
else
    cd self-improving-robots && git pull && cd ..
fi

echo "  - lerobot"
if [ ! -d "lerobot" ]; then
    git clone https://github.com/huggingface/lerobot.git
else
    cd lerobot && git pull && cd ..
fi

echo "  - robosuite"
if [ ! -d "robosuite" ]; then
    git clone https://github.com/ARISE-Initiative/robosuite.git
else
    cd robosuite && git pull && cd ..
fi

echo "  - ManiSkill"
if [ ! -d "maniskill" ]; then
    git clone https://github.com/haosulab/ManiSkill.git maniskill
else
    cd maniskill && git pull && cd ..
fi

echo ""
echo "✓ Repositories ready"
echo ""

# Step 3: Create/update conda environment
echo "Step 2: Creating conda environment: $ENV_NAME"
echo "  Python version: 3.11"
echo ""

# Create environment with Python 3.11
$CONDA_CMD create -n "$ENV_NAME" python=3.11 -y

# Get the activation script (micromamba uses different hook syntax)
if [ "$CONDA_CMD" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell bash)"
else
    eval "$($CONDA_CMD shell.bash hook)"
fi
$CONDA_CMD activate "$ENV_NAME"

echo "✓ Conda environment created and activated"
echo ""

# Step 4: Install core dependencies
echo "Step 3: Installing core dependencies..."

# Update pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU or GPU - adjust based on your server)
# For GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✓ PyTorch installed"
echo ""

# Step 5: Install lerobot
echo "Step 4: Installing lerobot..."
cd "$BASE_DIR/lerobot"
pip install -e .
cd "$BASE_DIR"
echo "✓ lerobot installed"
echo ""

# Step 6: Install robosuite (optional but recommended)
echo "Step 5: Installing robosuite..."
cd "$BASE_DIR/robosuite"
pip install -e .
cd "$BASE_DIR"
echo "✓ robosuite installed"
echo ""

# Step 7: Install ManiSkill
echo "Step 6: Installing ManiSkill..."
cd "$BASE_DIR/maniskill"
pip install -e .
cd "$BASE_DIR"
echo "✓ ManiSkill installed"
echo ""

# Step 8: Install self-improving-robots
echo "Step 7: Installing self-improving-robots..."
cd "$BASE_DIR/self-improving-robots"
pip install -e .
cd "$BASE_DIR"
echo "✓ self-improving-robots installed"
echo ""

# Step 9: Install additional training dependencies
echo "Step 8: Installing training dependencies..."
pip install \
    wandb \
    hydra-core \
    tensorboard \
    tqdm \
    pyyaml

echo "✓ Training dependencies installed"
echo ""

# Step 10: Verify installation
echo "Step 9: Verifying installation..."
python -c "
import torch
import lerobot
import gymnasium
import robosuite
import sir.training
import sir.teleoperation
print('✓ All imports successful!')
print(f'  PyTorch version: {torch.__version__}')
print(f'  Device available: {torch.backends.mps.is_available() or torch.cuda.is_available()}')
"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  $CONDA_CMD activate $ENV_NAME"
echo ""
echo "To submit a training job, run:"
echo "  sbatch scripts/slurm_train_act.sh --repo-id YOUR_REPO_ID"
echo ""
echo "Or use the provided template:"
echo "  bash scripts/slurm_submit_training.sh --repo-id YOUR_REPO_ID --job-name my_training"
echo ""
