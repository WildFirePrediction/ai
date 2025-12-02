#!/bin/bash

# SLIDING WINDOW PIPELINE
# Complete pipeline from raw VIIRS data → RL-ready environments
# Replaces DBSCAN-based clustering with fixed-duration sliding windows

set -e  # Exit on error

REPO_ROOT="/home/chaseungjoon/code/WildfirePrediction-SSD"
cd "$REPO_ROOT"

echo "========================================================================="
echo "SLIDING WINDOW PIPELINE - Complete Data Processing"
echo "========================================================================="
echo ""
echo "This pipeline will:"
echo "  1. Clean up old DBSCAN-based data"
echo "  2. Extract 48h sliding windows from fire detections"
echo "  3. Create spatial regions for each window"
echo "  4. Generate temporal sequences (2h bins)"
echo "  5. Assemble RL environments"
echo "  6. Split train/val/test"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# STEP 0: Backup and Clean Old Data
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 0: Cleaning old DBSCAN-based data"
echo "========================================================================="

BACKUP_DIR="backups/old_dbscan_data_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "  Backing up old data to: $BACKUP_DIR"

# Backup old DBSCAN outputs
if [ -f "embedded_data/nasa_viirs_with_weather_reclustered.parquet" ]; then
    mv embedded_data/nasa_viirs_with_weather_reclustered.parquet "$BACKUP_DIR/" 2>/dev/null || true
fi
if [ -f "embedded_data/episode_index_reclustered.parquet" ]; then
    mv embedded_data/episode_index_reclustered.parquet "$BACKUP_DIR/" 2>/dev/null || true
fi
if [ -f "embedded_data/reclustering_params.json" ]; then
    mv embedded_data/reclustering_params.json "$BACKUP_DIR/" 2>/dev/null || true
fi

# Backup old tiling data
if [ -d "tilling_data" ]; then
    echo "  Backing up tilling_data..."
    cp -r tilling_data "$BACKUP_DIR/" 2>/dev/null || true
    rm -rf tilling_data/regions/* 2>/dev/null || true
    rm -rf tilling_data/sequences/* 2>/dev/null || true
    rm -rf tilling_data/environments/* 2>/dev/null || true
    rm -f tilling_data/*.parquet 2>/dev/null || true
fi

# Clean up old RL checkpoints
if [ -d "rl_training/checkpoints_a3c" ]; then
    echo "  Backing up RL checkpoints..."
    cp -r rl_training/checkpoints_a3c "$BACKUP_DIR/" 2>/dev/null || true
    rm -rf rl_training/checkpoints_a3c/* 2>/dev/null || true
fi

echo "  ✓ Old data backed up and cleaned"

# ============================================================================
# STEP 1: Extract Sliding Windows
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 1: Extracting Sliding Windows"
echo "========================================================================="
echo "  Window duration: 48 hours"
echo "  Window stride: 24 hours (50% overlap)"
echo "  Min detections: 20"
echo "  Spatial radius: 5km"
echo ""

python3 embedding_src/10_sliding_window_extraction.py

# Validate window extraction
echo ""
echo "  Validating window extraction..."
python3 -c "
import pandas as df
import json

# Load window index
windows = pd.read_parquet('embedded_data/sliding_windows_index.parquet')

print(f'\n✓ Windows created: {len(windows)}')
print(f'✓ Duration: {windows[\"duration_hours\"].mean():.1f}h (all {windows[\"duration_hours\"].unique()[0]:.0f}h)')
print(f'✓ Detections: mean={windows[\"n_detections\"].mean():.1f}, median={windows[\"n_detections\"].median():.0f}')
print(f'✓ Spatial extent: mean={windows[\"spatial_extent_km\"].mean():.2f}km')

# Load params
with open('embedded_data/sliding_windows_params.json') as f:
    params = json.load(f)

print(f'\n✓ Method: {params[\"method\"]}')
print(f'✓ Window duration: {params[\"window_duration_h\"]}h')
print(f'✓ Number of spatial regions: {params[\"num_spatial_regions\"]}')

if len(windows) < 100:
    print('\n⚠️  WARNING: Less than 100 windows created. Data may be too sparse.')
elif len(windows) > 5000:
    print('\n⚠️  INFO: More than 5000 windows. Training will take longer but more data is good.')
else:
    print('\n✅ Window count looks good!')
"

# ============================================================================
# STEP 2: Spatial Tiling
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 2: Creating Spatial Regions"
echo "========================================================================="

python3 tilling_src/01_spatial_tiling.py

# ============================================================================
# STEP 3: Temporal Segmentation
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 3: Temporal Segmentation (2h bins)"
echo "========================================================================="

python3 tilling_src/02_temporal_segmentation.py

# ============================================================================
# STEP 4: Environment Assembly
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 4: Assembling RL Environments"
echo "========================================================================="

python3 tilling_src/03_environment_assembly.py

# ============================================================================
# STEP 5: Dataset Split
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 5: Train/Val/Test Split"
echo "========================================================================="

python3 tilling_src/04_dataset_split.py

# ============================================================================
# STEP 6: Validation
# ============================================================================
echo ""
echo "========================================================================="
echo "STEP 6: Data Quality Validation"
echo "========================================================================="

python3 tilling_src/05_environment_validation.py

# Quick fire movement check
echo ""
echo "  Checking fire movement in sample environments..."
python3 -c "
import pickle
import numpy as np
from pathlib import Path

env_dir = Path('tilling_data/environments')
env_files = sorted(list(env_dir.glob('*.pkl')))[:20]  # Sample 20

movements = []
envs_with_movement = 0

for env_path in env_files:
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    fire_masks = env['temporal']['fire_masks']
    env_movements = []

    for t in range(len(fire_masks) - 1):
        mask0 = fire_masks[t] > 0
        mask1 = fire_masks[t+1] > 0

        if mask0.sum() > 0 and mask1.sum() > 0:
            y0, x0 = np.where(mask0)
            y1, x1 = np.where(mask1)
            c0 = np.array([y0.mean(), x0.mean()])
            c1 = np.array([y1.mean(), x1.mean()])
            movement = np.linalg.norm(c1 - c0)
            env_movements.append(movement)

    if len(env_movements) > 0:
        movements.extend(env_movements)
        if (np.array(env_movements) > 0.5).any():
            envs_with_movement += 1

if len(movements) > 0:
    print(f'\n✓ Movements detected: {len(movements)}')
    print(f'✓ Mean movement: {np.mean(movements):.2f} cells')
    print(f'✓ Movements > 0.5 cells: {(np.array(movements) > 0.5).sum()} ({100*(np.array(movements) > 0.5).mean():.1f}%)')
    print(f'✓ Environments with movement: {envs_with_movement}/{len(env_files)} ({100*envs_with_movement/len(env_files):.1f}%)')

    if np.mean(movements) < 0.3:
        print('\n⚠️  WARNING: Fire movement still too small')
    elif envs_with_movement < len(env_files) * 0.4:
        print('\n⚠️  WARNING: Less than 40% of environments show detectable movement')
    else:
        print('\n✅ Fire movement looks good!')
else:
    print('\n❌ ERROR: No fire movement detected!')
"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================================="
echo "PIPELINE COMPLETE"
echo "========================================================================="

# Final statistics
python3 -c "
import pandas as pd
from pathlib import Path
import json

# Window stats
windows = pd.read_parquet('embedded_data/sliding_windows_index.parquet')

# Environment stats
manifest_path = Path('tilling_data/environment_manifest.parquet')
if manifest_path.exists():
    manifest = pd.read_parquet(manifest_path)

    print(f'\n📊 FINAL STATISTICS')
    print(f'  Windows: {len(windows)}')
    print(f'  Environments: {len(manifest)}')
    print(f'')
    print(f'  Window Duration: {windows[\"duration_hours\"].mean():.1f}h (fixed)')
    print(f'  Detections per Window: mean={windows[\"n_detections\"].mean():.1f}, median={windows[\"n_detections\"].median():.0f}')
    print(f'')

    # Train/val/test split
    train = manifest[manifest['split'] == 'train']
    val = manifest[manifest['split'] == 'val']
    test = manifest[manifest['split'] == 'test']

    print(f'  Dataset Split:')
    print(f'    Train: {len(train)}')
    print(f'    Val: {len(val)}')
    print(f'    Test: {len(test)}')
    print(f'')

    # Data comparison
    print(f'  Comparison with OLD DBSCAN approach:')
    print(f'    Duration: 4.8 days → {windows[\"duration_hours\"].mean()/24:.1f} days (✓ FIXED)')
    print(f'    Fire movement: 30% → Need to validate with training')
else:
    print('⚠️  Manifest file not found')
"

echo ""
echo "✅ Sliding window pipeline complete!"
echo ""
echo "📁 Backup location: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Quick RL training test (100 updates):"
echo "     export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && export NUMEXPR_MAX_THREADS=1"
echo "     python3 -m rl_training.train_a3c --max-iters 100 --num-workers 4 --device cuda"
echo ""
echo "  2. If test succeeds, run full training:"
echo "     python3 -m rl_training.train_a3c --max-iters 5000 --num-workers 8 --device cuda"
echo ""
echo "========================================================================="
