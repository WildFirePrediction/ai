# Wildfire Prediction Training Progress

## Summary
Training spatial wildfire prediction models using tilled satellite data. Abandoned A3C due to poor performance, now focusing on supervised learning with U-Net.

---

## Model Architecture Evolution

### A3C (Abandoned)
- **Status**: Abandoned due to poor performance
- **Architecture**: Actor-Critic with spatial fire spread prediction
- **Issues**:
  - Poor convergence
  - Inefficient training with RL approach
  - Decided supervised learning would be more effective

### U-Net (Current Focus)

#### Version 1: Original (Too Large)
- **Parameters**: 31M
- **Architecture**:
  - Encoder: 64 → 128 → 256 → 512 channels
  - Decoder: 512 → 256 → 128 → 64 channels
  - 4 down/up layers
- **Status**: Failed - GPU memory errors (CUDNN_STATUS_INTERNAL_ERROR)

#### Version 2: Reduced (Current)
- **Parameters**: 1.9M (reduced from 31M)
- **Architecture**:
  - Encoder: 32 → 64 → 128 → 256 channels (reduced by half)
  - Decoder: 256 → 128 → 64 → 32 channels
  - 3 down/up layers (removed one layer)
- **Status**: Fits in GPU memory, training possible
- **Location**: `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/supervised/unet_model.py`

**Model Details**:
```python
# Input: (B, 14, H, W) - environmental features
# Output: (B, 1, H, W) - burn probability per cell

self.inc = DoubleConv(14, 32)
self.down1 = Down(32, 64)
self.down2 = Down(64, 128)
self.down3 = Down(128, 256)
self.up1 = Up(256, 128)
self.up2 = Up(128, 64)
self.up3 = Up(64, 32)
self.outc = Conv2d(32, 1, kernel_size=1)
```

---

## Training Configuration Evolution

### Batch Size Attempts
1. **Batch size 4**: CUDNN_STATUS_INTERNAL_ERROR (GPU OOM)
2. **Batch size 2**: CUDNN_STATUS_INTERNAL_ERROR (GPU OOM)
3. **Batch size 1**: CUDNN_STATUS_INTERNAL_ERROR (GPU OOM)
4. **After model reduction, Batch size 2**: SUCCESS (fits in memory)

### Current Training Config
- **Epochs**: 5 (changed from 50 for testing)
- **Batch size**: 2
- **Learning rate**: 1e-4
- **Optimizer**: Adam
- **Loss**: Binary Cross-Entropy with Logits
- **Metric**: IoU (Intersection over Union)
- **DataLoader workers**: 4
- **Device**: CUDA

### Checkpointing Strategy Changes

#### Original Strategy
- Best model: Saved when validation IoU improves
- Regular checkpoints: Every 10 epochs

#### Current Strategy (Modified)
- Best model: Saved when validation IoU improves
- Regular checkpoints: **EVERY epoch** (changed per user request)
- Location: `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/supervised/checkpoints_unet/`
- Files:
  - `best_model.pt` - Best validation IoU
  - `checkpoint_epoch{N}.pt` - Every epoch checkpoint

**Modified Code** (`train_unet.py` lines 262-282):
```python
# Save best model
if val_iou > best_val_iou:
    best_val_iou = val_iou
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'val_loss': val_loss
    }, checkpoint_dir / 'best_model.pt')
    print(f"✓ Saved best model (IoU: {val_iou:.4f})")

# Save checkpoint EVERY epoch (CHANGED FROM: if epoch % 10 == 0)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_iou': val_iou,
    'val_loss': val_loss
}, checkpoint_dir / f'checkpoint_epoch{epoch}.pt')
print(f"✓ Saved checkpoint for epoch {epoch}")
```

---

## Dataset Details

### Location
- **Directory**: `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_data/environments/`
- **Format**: Pickle files (`.pkl`)
- **Split files**:
  - `train_split.json` - 3026 training environments
  - `val_split.json` - 648 validation environments

### Dataset Statistics
- **Training samples**: 23,945 (multiple timesteps per environment)
- **Validation samples**: ~5,000 (estimate)
- **Input channels**: 14 (environmental features)
- **Output**: Binary mask (1 channel) - which cells will burn next

### Custom DataLoader Features
- **Variable spatial dimensions**: Different environments have different H×W sizes
- **Custom collate_fn**: Pads all samples in batch to max dimensions
- **Padding**: Zero-padding for both observations and targets

---

## Critical Dataset Problem (Current Blocker)

### Issue: Extremely Large Environment Files
Training hangs during dataset initialization ("Scanning envs" phase) at 97% complete (2939/3026 environments).

### Root Cause
Some environment files are **500-1000x larger** than typical files:

**Normal files**: 88-114 KB
**Problem files**:
- `env_02392.pkl` - **75 MB** (where training stuck)
- `env_03524.pkl` - **74 MB**
- `env_03671.pkl` - **57 MB**
- `env_04269.pkl` - **34 MB**
- `env_04270.pkl` - **64 MB**
- `env_04271.pkl` - **59 MB**
- `env_04276.pkl` - **91 MB**
- `env_04277.pkl` - **81 MB**
- Many more 30-90 MB files

### Symptoms
- Process stuck at 2939/3026 environments (97%)
- GPU usage at 99% during file loading (ABNORMAL - should be CPU-only)
- Frozen for 4+ minutes with no progress
- Process needs to be killed

### Training Split Position
The stuck position (2939) corresponds to:
- 2937: env_03524 (74 MB)
- 2938: env_03671 (57 MB)
- **2939: env_02392 (75 MB)** ← STUCK HERE
- 2940: env_00527 (16 KB - normal)

---

## Options and Next Steps

### Option 1: Filter Out Large Files
**Pros**:
- Quick fix
- Training can proceed immediately
- Still have 3000+ environments

**Cons**:
- Lose potentially valuable data
- Unknown why these files are large

**Implementation**:
```python
# Add file size check in dataset loading
import os
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
valid_paths = [p for p in env_paths if os.path.getsize(p) < MAX_FILE_SIZE]
```

### Option 2: Investigate Why Files Are Large
**Pros**:
- Understand root cause
- May reveal data quality issues
- Could fix at source

**Cons**:
- Takes time to debug
- May not be fixable

**Investigation steps**:
1. Load one large file and inspect contents
2. Compare with normal file structure
3. Check if it's a bug in data generation pipeline
4. Potentially regenerate these environments

### Option 3: Lazy Loading with Timeout
**Pros**:
- Handle problematic files gracefully
- Automatic recovery

**Cons**:
- Adds complexity
- May still cause issues during actual training

**Implementation**:
```python
# Add timeout and error handling in __getitem__
try:
    with timeout(seconds=10):
        env = WildfireEnvSpatial(env_path)
except TimeoutError:
    # Skip this sample
    return self.__getitem__((idx + 1) % len(self))
```

### Option 4: Pre-validate Dataset
**Pros**:
- One-time validation
- Clean dataset going forward
- Can document problematic files

**Cons**:
- Need to run validation script first
- May take time

**Implementation**:
Create script to:
1. Test load all environment files
2. Record file size, load time, structure
3. Generate blacklist of problematic files
4. Update train/val splits to exclude them

---

## Training Estimates (When Unblocked)

### Per Epoch (with batch_size=2)
- **Time**: 1.7 - 3.3 hours per epoch
- **Iterations**: ~12,000 training batches
- **Memory**: Fits in GPU with current model

### Full Training (50 epochs)
- **Total time**: 3.5 - 7 days
- **Total time (5 epochs)**: 8.5 - 16.5 hours

---

## File Locations Summary

### Model Files
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/supervised/unet_model.py` - U-Net architecture
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/supervised/train_unet.py` - Training script
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/wildfire_env_spatial.py` - Environment class

### Data Files
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_data/environments/*.pkl` - Environment files
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_data/environments/train_split.json` - Train split
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_data/environments/val_split.json` - Val split

### Checkpoint Directory
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/supervised/checkpoints_unet/`

---

## Command to Resume Training

```bash
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 rl_training/supervised/train_unet.py \
  --epochs 5 \
  --batch-size 2 \
  --lr 1e-4 \
  --num-workers 4
```

**Note**: Will hang at 97% until dataset issue is resolved.

---

## Related Work (Context)

### A3C Training Files (Abandoned but Still Present)
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/a3c/model.py` - A3C model
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/a3c/train.py` - A3C training
- `/home/chaseungjoon/code/WildfirePrediction-SSD/rl_training/a3c/worker.py` - A3C workers

### Data Generation Pipeline
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_src/01_spatial_tiling.py`
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_src/02_temporal_segmentation.py`
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_src/03_environment_assembly.py`
- `/home/chaseungjoon/code/WildfirePrediction-SSD/tilling_src/04_dataset_split.py`

---

## Key Decisions Made

1. **Abandoned A3C**: RL approach not suitable for this problem
2. **Chose U-Net**: Better for dense spatial prediction tasks
3. **Reduced model size**: From 31M to 1.9M parameters to fit GPU
4. **Batch size 2**: Optimal balance between memory and training speed
5. **Save every epoch**: For fine-grained checkpoint recovery
6. **5 epoch test**: Before committing to full 50-epoch training

---

## Next Session TODO

1. **URGENT**: Fix dataset loading issue
   - Choose one of the 4 options above
   - Recommended: Option 1 (filter) + Option 2 (investigate) in parallel

2. **Resume training**: Once dataset is fixed, run 5-epoch test

3. **Monitor training**: Check for:
   - GPU memory stability
   - Training speed matches estimates
   - Loss/IoU trends look reasonable

4. **If 5 epochs succeed**: Scale up to 50 epochs

5. **Evaluate results**: Check validation IoU and compare to baseline

---

## Questions to Answer Later

1. Why are some environment files 75 MB vs 100 KB?
2. Is there a bug in the data generation pipeline?
3. Should we regenerate the problematic environments?
4. What's causing GPU usage during pickle file loading?
5. Are these large files even valid training data?

---

Last Updated: 2025-11-09
Status: BLOCKED on dataset loading issue
Next Action: Fix large environment file problem
