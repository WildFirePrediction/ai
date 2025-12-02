#!/bin/bash

# Fire Data Rebuild Automation Script
# This script rebuilds fire episode clustering and tiling with optimized parameters for RL training
# See FIRE_DATA_REBUILD_PLAN.md for details

set -e  # Exit on error

REPO_ROOT="/home/chaseungjoon/code/WildfirePrediction-SSD"
cd "$REPO_ROOT"

echo "========================================================================"
echo "FIRE DATA REBUILD PIPELINE"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Backup old fire data"
echo "  2. Re-cluster episodes with stricter parameters (2 days, min 10 detections)"
echo "  3. Re-tile with 2-hour bins (was 6-hour)"
echo "  4. Validate new data quality"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# STEP 0: Backup old data
# ============================================================================
echo ""
echo "STEP 0: Backing up old data..."
BACKUP_DIR="backups/old_fire_data_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup fire clustering outputs
if [ -f "embedded_data/nasa_viirs_with_weather_reclustered.parquet" ]; then
    echo "  Backing up reclustered fire data..."
    cp embedded_data/nasa_viirs_with_weather_reclustered.parquet "$BACKUP_DIR/"
    cp embedded_data/episode_index_reclustered.parquet "$BACKUP_DIR/"
    cp embedded_data/reclustering_params.json "$BACKUP_DIR/" 2>/dev/null || true
fi

# Backup tiling outputs
if [ -d "tilling_data" ]; then
    echo "  Backing up tiling data..."
    cp -r tilling_data "$BACKUP_DIR/"
fi

# Backup RL checkpoints
if [ -d "rl_training/checkpoints_a3c" ]; then
    echo "  Backing up RL checkpoints..."
    cp -r rl_training/checkpoints_a3c "$BACKUP_DIR/"
fi

echo "  ✓ Backup saved to: $BACKUP_DIR"

# ============================================================================
# STEP 1: Delete old fire data
# ============================================================================
echo ""
echo "STEP 1: Deleting old fire data..."

# Delete reclustered fire data
rm -f embedded_data/nasa_viirs_with_weather_reclustered.parquet
rm -f embedded_data/episode_index_reclustered.parquet
rm -f embedded_data/reclustering_params.json

# Delete all tiling outputs
rm -rf tilling_data/regions/
rm -rf tilling_data/sequences/
rm -rf tilling_data/environments/
rm -f tilling_data/episode_regions.parquet
rm -f tilling_data/environment_manifest.parquet
rm -f tilling_data/*.json

# Delete RL checkpoints
rm -rf rl_training/checkpoints_a3c/*

echo "  ✓ Old data deleted"

# ============================================================================
# STEP 2: Re-cluster fire episodes
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 2: Re-clustering fire episodes"
echo "========================================================================"
echo "Parameters:"
echo "  - Spatial threshold: 2km (unchanged)"
echo "  - Temporal threshold: 2 days (was 7 days)"
echo "  - Min samples: 10 (was 3)"
echo ""

python3 embedding_src/09_recluster_fire_episodes.py

# Validate clustering results
echo ""
echo "Validating clustering results..."
python3 -c "
import pandas as pd

df = pd.read_parquet('embedded_data/episode_index_reclustered.parquet')
print(f'\n✓ Episodes created: {len(df)}')
print(f'✓ Duration: mean={df[\"duration_hours\"].mean():.1f}h, median={df[\"duration_hours\"].median():.1f}h, max={df[\"duration_hours\"].max():.1f}h')
print(f'✓ Detections: mean={df[\"n_detections\"].mean():.1f}, median={df[\"n_detections\"].median():.0f}')

long_eps = (df['duration_hours'] > 48).sum()
print(f'✓ Episodes > 48h: {long_eps} ({100*long_eps/len(df):.1f}%)')

sparse_eps = (df['n_detections'] < 20).sum()
print(f'✓ Episodes < 20 dets: {sparse_eps} ({100*sparse_eps/len(df):.1f}%)')

if long_eps > len(df) * 0.2:
    print('\n⚠️  WARNING: More than 20% of episodes exceed 48h duration')
if sparse_eps > len(df) * 0.3:
    print('⚠️  WARNING: More than 30% of episodes have < 20 detections')
"

# ============================================================================
# STEP 3: Re-tile with 2-hour bins
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 3: Re-tiling with 2-hour temporal bins"
echo "========================================================================"
echo "Parameters:"
echo "  - Temporal resolution: 2 hours (was 6 hours)"
echo "  - Temporal window: ±1 hour (was ±3 hours)"
echo ""

echo "[3.1] Spatial tiling..."
python3 tilling_src/01_spatial_tiling.py

echo ""
echo "[3.2] Temporal segmentation..."
python3 tilling_src/02_temporal_segmentation.py

echo ""
echo "[3.3] Environment assembly..."
python3 tilling_src/03_environment_assembly.py

echo ""
echo "[3.4] Dataset split..."
python3 tilling_src/04_dataset_split.py

# ============================================================================
# STEP 4: Validate new data
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 4: Validating new data quality"
echo "========================================================================"

python3 tilling_src/05_environment_validation.py

# Check fire movement in sample environments
echo ""
echo "Checking fire movement in sample environments..."
python3 -c "
import pickle
import numpy as np
from pathlib import Path

env_dir = Path('tilling_data/environments')
env_files = sorted(list(env_dir.glob('*.pkl')))[:10]  # Sample 10 environments

total_movements = []
total_envs_with_movement = 0

for env_path in env_files:
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    fire_masks = env['temporal']['fire_masks']
    movements = []

    for t in range(len(fire_masks) - 1):
        mask0 = fire_masks[t] > 0
        mask1 = fire_masks[t+1] > 0

        if mask0.sum() > 0 and mask1.sum() > 0:
            y0, x0 = np.where(mask0)
            y1, x1 = np.where(mask1)
            c0 = np.array([y0.mean(), x0.mean()])
            c1 = np.array([y1.mean(), x1.mean()])
            movement = np.linalg.norm(c1 - c0)
            movements.append(movement)

    if len(movements) > 0:
        total_movements.extend(movements)
        if (np.array(movements) > 0.5).any():
            total_envs_with_movement += 1

if len(total_movements) > 0:
    print(f'\n✓ Total movements detected: {len(total_movements)}')
    print(f'✓ Mean movement magnitude: {np.mean(total_movements):.2f} cells')
    print(f'✓ Movements > 0.5 cells: {(np.array(total_movements) > 0.5).sum()} ({100*(np.array(total_movements) > 0.5).mean():.1f}%)')
    print(f'✓ Environments with detectable movement: {total_envs_with_movement}/{len(env_files)} ({100*total_envs_with_movement/len(env_files):.1f}%)')

    if np.mean(total_movements) < 0.3:
        print('\n⚠️  WARNING: Fire movement is still too small')
    elif total_envs_with_movement < len(env_files) * 0.5:
        print('\n⚠️  WARNING: Less than 50% of environments show detectable movement')
    else:
        print('\n✅ Fire movement looks good!')
else:
    print('\n❌ ERROR: No fire movement detected in sample environments!')
"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "REBUILD COMPLETE"
echo "========================================================================"

# Get final statistics
python3 -c "
import pandas as pd
from pathlib import Path

# Episode stats
episodes = pd.read_parquet('embedded_data/episode_index_reclustered.parquet')

# Environment stats
manifest_path = Path('tilling_data/environment_manifest.parquet')
if manifest_path.exists():
    manifest = pd.read_parquet(manifest_path)

    print(f'\n📊 FINAL STATISTICS')
    print(f'  Episodes: {len(episodes)}')
    print(f'  Environments: {len(manifest)}')
    print(f'')
    print(f'  Episode Duration:')
    print(f'    Mean: {episodes[\"duration_hours\"].mean():.1f}h')
    print(f'    Median: {episodes[\"duration_hours\"].median():.1f}h')
    print(f'    Max: {episodes[\"duration_hours\"].max():.1f}h')
    print(f'')
    print(f'  Detections per Episode:')
    print(f'    Mean: {episodes[\"n_detections\"].mean():.1f}')
    print(f'    Median: {episodes[\"n_detections\"].median():.0f}')
    print(f'')

    # Train/val/test split
    train = manifest[manifest['split'] == 'train']
    val = manifest[manifest['split'] == 'val']
    test = manifest[manifest['split'] == 'test']

    print(f'  Dataset Split:')
    print(f'    Train: {len(train)}')
    print(f'    Val: {len(val)}')
    print(f'    Test: {len(test)}')
else:
    print('⚠️  Manifest file not found')
"

echo ""
echo "✅ Fire data rebuild complete!"
echo ""
echo "📁 Backup location: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Test RL training with new data:"
echo "     python3 -m rl_training.train_a3c --max-iters 100 --num-workers 4 --device cuda"
echo ""
echo "  2. If results look good, run full training:"
echo "     python3 -m rl_training.train_a3c --max-iters 5000 --num-workers 8 --device cuda"
echo ""
echo "========================================================================"
