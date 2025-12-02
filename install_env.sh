#!/bin/bash

set -e

echo "=== 1. Remove old NVIDIA/CUDA packages ==="
sudo systemctl stop display-manager || true
sudo apt remove --purge '^nvidia-.*' cuda* libcudnn* --yes || true
sudo apt autoremove --yes || true
sudo apt autoclean || true

echo "=== Rebooting required… please reboot manually after this script if display-manager was stopped ==="
sleep 2

echo "=== 2. Install NVIDIA Driver 580.95.05 ==="
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-580

echo "=== 3. Download CUDA 13.0 ==="
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_560.35.03_linux.run -O cuda13.run
chmod +x cuda13.run

echo "=== 4. Install CUDA Toolkit 13.0 (no driver) ==="
sudo sh cuda13.run --silent --toolkit --override

echo "=== 5. Add CUDA to PATH ==="
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
fi
if ! grep -q "/usr/local/cuda/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
source ~/.bashrc

echo "=== 6. Check NVIDIA Driver ==="
nvidia-smi || echo "NVIDIA driver not active yet (reboot required)"

echo "=== 7. Check CUDA version ==="
nvcc --version || echo "nvcc not found (PATH issue or reboot required)"

echo "=== Installation complete! Reboot strongly recommended. ==="
