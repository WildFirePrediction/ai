# A3C V3.5 COLAB Setup Guide - FULL POWER MODE

## 🚀 Google Colab with Per-Pixel LSTM (Architecture Plan Implementation)

This is the FULL POWER version of V3.5 that follows the architecture plan **WITHOUT COMPROMISES**:
- ✅ **5 timestep window** (architecture plan spec)
- ✅ **4-8 workers** (parallel training)
- ✅ **150K grid cells** (387×387 grids)
- ✅ **Per-pixel LSTM** (2 layers, hidden=128)
- ✅ **GPU-accelerated** (V100/A100)

## Prerequisites

1. **Google Colab Account** (free or Pro)
2. **Recommended**: Colab Pro for High-RAM runtime (25GB RAM)
3. **WandB Account** (optional, for logging)

---

## Step 1: Create New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

---

## Step 2: Setup Runtime

**CRITICAL**: Set runtime to **High-RAM + GPU**

```
Runtime → Change runtime type → Hardware accelerator: GPU → Runtime shape: High-RAM
```

Verify you have enough resources:
```python
# Check RAM
!free -h
# Should show ~25GB total

# Check GPU
!nvidia-smi
# Should show V100 or A100 with ~16GB VRAM
```

---

## Step 3: Mount Google Drive (if your data is there)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 4: Clone Repository

```python
# Clone your WildfirePrediction repo
!git clone https://github.com/YOUR_USERNAME/WildfirePrediction.git
%cd WildfirePrediction
```

**Or** if you already have it on Drive:
```python
import os
os.chdir('/content/drive/MyDrive/WildfirePrediction')
```

---

## Step 5: Install Dependencies

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install wandb tqdm numpy scipy
```

---

## Step 6: Upload Data (if not on Drive)

```python
# Option A: Upload from local machine
from google.colab import files
uploaded = files.upload()  # Upload your tilling_data folder

# Option B: Download from cloud storage
!wget YOUR_DATA_URL -O data.zip
!unzip data.zip
```

---

## Step 7: Login to WandB (optional)

```python
import wandb
wandb.login()  # Will prompt for API key
```

---

## Step 8: Run Training (FULL POWER)

### Default Configuration (4 workers, 5 timesteps)

```python
!PYTHONPATH=/content/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v3_5_colab.py \
  --repo-root /content/WildfirePrediction \
  --max-episodes 1000 \
  --num-workers 4 \
  --temporal-window 5 \
  --max-grid-cells 150000 \
  --min-episode-length 4 \
  --wandb-project wildfire-prediction-colab \
  --wandb-run-name "v3.5-colab-full-power"
```

### High Performance (8 workers, aggressive)

```python
!PYTHONPATH=/content/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v3_5_colab.py \
  --repo-root /content/WildfirePrediction \
  --max-episodes 2000 \
  --num-workers 8 \
  --temporal-window 5 \
  --max-grid-cells 150000 \
  --min-episode-length 4 \
  --lr 7e-5 \
  --entropy-coef 0.015 \
  --wandb-project wildfire-prediction-colab \
  --wandb-run-name "v3.5-colab-8workers-2k-episodes"
```

### Test Run (quick verification)

```python
!PYTHONPATH=/content/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v3_5_colab.py \
  --repo-root /content/WildfirePrediction \
  --max-episodes 10 \
  --num-workers 2 \
  --max-envs 100 \
  --no-wandb
```

---

## Step 9: Monitor Training

### In Notebook (Real-time)

The training script will print logs directly in the notebook cell output.

### WandB Dashboard

Go to your WandB project page: `https://wandb.ai/YOUR_USERNAME/wildfire-prediction-colab`

### GPU Monitoring

Open a new cell and run:
```python
!watch -n 1 nvidia-smi
```

---

## Step 10: Download Trained Model

After training completes:

```python
# Find the checkpoint directory
!ls -lh rl_training/a3c/checkpoints_v3_5_colab/

# Download best model
from google.colab import files
files.download('rl_training/a3c/checkpoints_v3_5_colab/TIMESTAMP/best_model.pt')
files.download('rl_training/a3c/checkpoints_v3_5_colab/TIMESTAMP/final_model.pt')
```

---

## Expected Performance

### Colab Free (T4 GPU, ~12GB RAM)
- **Workers**: 2-3
- **Temporal window**: 5
- **Max grid**: 100K cells
- **Training time**: ~2-3 hours for 1000 episodes

### Colab Pro (V100/A100, ~25GB RAM)
- **Workers**: 4-8
- **Temporal window**: 5
- **Max grid**: 150K cells
- **Training time**: ~1-2 hours for 1000 episodes

### Expected IoU
- **Target**: 47-50% (architecture plan goal)
- **V3 baseline**: 40.91% IoU
- **Expected improvement**: +6-9% from temporal modeling

---

## Troubleshooting

### OOM (Out of Memory)

If you get OOM even on Colab:

```python
# Reduce workers
--num-workers 2

# Reduce grid size
--max-grid-cells 100000

# Reduce temporal window (not recommended, but works)
--temporal-window 3
```

### Slow Training

```python
# Use fewer environments for faster scanning
--max-envs 500

# Relax episode quality filter
--min-episode-length 3

# Increase learning rate slightly
--lr 1e-4
```

### Colab Disconnects

Colab free tier disconnects after ~12 hours. To prevent data loss:

1. **Save checkpoints frequently**: Already done every 50 episodes
2. **Use WandB**: Automatically syncs metrics
3. **Colab Pro**: 24-hour sessions

**To resume training** after disconnect:
```python
# Load checkpoint and continue
--resume-from /content/WildfirePrediction/rl_training/a3c/checkpoints_v3_5_colab/TIMESTAMP/best_model.pt
```

---

## Command Line Arguments (All Options)

```
--repo-root                Path to WildfirePrediction repo (default: /content/WildfirePrediction)
--num-workers              Number of parallel workers (default: 4, range: 2-8)
--max-episodes             Total episodes to train (default: 1000)
--temporal-window          Temporal window size (default: 5)
--max-grid-cells           Max grid size in cells (default: 150000 = 387×387)
--min-episode-length       Min timesteps with burns (default: 4)
--lr                       Learning rate (default: 7e-5)
--gamma                    Discount factor (default: 0.99)
--value-loss-coef          Value loss coefficient (default: 0.5)
--entropy-coef             Entropy coefficient (default: 0.015)
--max-grad-norm            Max gradient norm (default: 0.5)
--seed                     Random seed (default: 42)
--max-envs                 Limit training environments (default: all)
--max-file-size-mb         Skip large files (default: 50MB)
--log-interval             Log every N episodes (default: 10)
--wandb-project            WandB project name (default: wildfire-prediction-colab)
--wandb-run-name           WandB run name (default: auto-generated)
--no-wandb                 Disable WandB logging
--mixed-precision          Use mixed precision (experimental)
```

---

## Architecture Details

### Model: Per-Pixel LSTM

```
Input: (Batch, 5 timesteps, 14 channels, H, W)
  ↓
CNN Encoder (per timestep):
  14 → 32 → 64 → 128 channels
  ↓
Reshape to per-pixel sequences:
  (B*H*W, 5, 128)
  ↓
LSTM (2 layers, hidden=128):
  Processes each pixel's temporal evolution
  Learns: velocity, acceleration, wind dynamics
  ↓
Reshape back to spatial:
  (B, 128, H, W)
  ↓
Policy Head (per burning cell):
  Predict 8-neighbor spread
  ↓
Value Head (global):
  Predict state value
```

### Memory Profile (Colab)

| Component | Memory | Notes |
|-----------|--------|-------|
| Environment data | 2GB | Per worker |
| LSTM forward pass | 200MB | With chunking |
| Recomputation loop | 1.5GB | Per episode |
| PyTorch buffers | 1GB | Gradients + optimizer |
| **Per worker** | **~5GB** | Total |
| **4 workers** | **~20GB** | Fits in 25GB! |
| **GPU model** | **1GB** | VRAM usage |

---

## Comparison: Local vs Colab

| Parameter | Local (Emergency) | Colab (Full Power) |
|-----------|-------------------|---------------------|
| **Temporal window** | 3 | **5** ✅ |
| **Workers** | 1 | **4-8** ✅ |
| **Max grid** | 50K cells | **150K cells** ✅ |
| **Training time** | 2-3 hours | **1-2 hours** ✅ |
| **Follows plan** | ❌ Compromised | ✅ **100%** |

---

## Quick Start (Copy-Paste)

```python
# 1. Setup
!git clone https://github.com/YOUR_USERNAME/WildfirePrediction.git
%cd WildfirePrediction
!pip install -q torch torchvision torchaudio wandb tqdm numpy scipy

# 2. Login to WandB (optional)
import wandb
wandb.login()

# 3. Train (FULL POWER)
!PYTHONPATH=/content/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v3_5_colab.py \
  --repo-root /content/WildfirePrediction \
  --max-episodes 1000 \
  --num-workers 4 \
  --temporal-window 5 \
  --wandb-project wildfire-prediction-colab

# 4. Download model
from google.colab import files
!ls rl_training/a3c/checkpoints_v3_5_colab/
# files.download('rl_training/a3c/checkpoints_v3_5_colab/TIMESTAMP/best_model.pt')
```

---

## 🎯 Expected Results

With Colab's full resources and 5 timestep window:

- **Target IoU**: 47-50% (architecture plan goal)
- **Training time**: 1-2 hours for 1000 episodes
- **V3 baseline**: 40.91% IoU
- **Expected improvement**: +6-9% from temporal LSTM

**This is the TRUE V3.5 as specified in the architecture plan!** 🚀

---

## Support

If you encounter issues:

1. Check Colab runtime type (High-RAM + GPU)
2. Verify GPU allocation: `!nvidia-smi`
3. Check RAM usage: `!free -h`
4. Review WandB logs for training issues
5. Reduce workers/grid size if OOM

---

**Last Updated**: 2025-11-18
**Version**: V3.5 Colab Full Power
