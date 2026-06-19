#!/bin/bash
# Run this interactively on the cluster login node

echo "=== R2P-GEN Environment Setup ==="

# Load modules
module load cuda/12.4
module load python/3.12

# Create venv
echo "Creating virtual environment..."
python -m venv ~/venvs/r2p-gen
source ~/venvs/r2p-gen/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch first (CUDA 12.1 compatible with 12.4)
echo "Installing PyTorch..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn from wheel URL
echo "Installing flash-attn..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install CLIP from git
echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Install main requirements (skip torch and flash-attn since already installed)
echo "Installing requirements..."
cd ~/R2P-GEN
grep -v "^torch==" requirements.txt | grep -v "^flash-attn" | grep -v "^#" | pip install -r /dev/stdin

# Upgrade transformers for Qwen2-VL
echo "Upgrading transformers for Qwen2-VL..."
pip install "transformers>=4.45.0"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source ~/venvs/r2p-gen/bin/activate"