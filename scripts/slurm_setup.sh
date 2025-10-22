#!/bin/bash
#
# SLURM Server Setup Script
#
# This script sets up a complete environment for training on a SLURM-based cluster.
# Usage: bash scripts/slurm_setup.sh [--base-dir BASE_DIR] [--env-name ENV_NAME]
#
# The script will:
# 1. Clone/update all required repositories
# 2. Create a conda environment with all dependencies
# 3. Install the self-improving-robots package in editable mode
# 4. Verify the installation
#

set -e  # Exit on any error

# Configuration
BASE_DIR="${1:-.}"
ENV_NAME="${2:-sir}"
CONDA_INIT_FILE=""

# Detect conda/mamba command
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: Neither mamba nor conda found. Please install one of them first."
    echo "  mamba: https://github.com/conda-forge/miniforge"
    echo "  conda: https://www.anaconda.com/download"
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
    git clone https://github.com/lerobotics/lerobot.git
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

# Get the conda activation script
eval "$($CONDA_CMD shell.bash hook)"
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
from sir.training import train_act
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
