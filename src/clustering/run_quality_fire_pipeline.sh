#!/bin/bash
#
# Complete Pipeline for Quality Fire Data
# Filters for long-duration fires (5+ detections, 3+ days) and rebuilds all data
#

set -e  # Exit on error

REPO_ROOT="/home/chaseungjoon/code/WildfirePrediction"
cd "$REPO_ROOT"

# Activate virtual environment
source .venv/bin/activate

echo "============================================================"
echo " QUALITY FIRE DATA PIPELINE"
echo "============================================================"
echo

# ============================================================
# Step 1: Filter VIIRS data for quality fires
# ============================================================
echo "[1/6] Filtering VIIRS data for long-duration fires..."
python3 tilling_src/01_filter_quality_fires.py
if [ $? -ne 0 ]; then
    echo "ERROR: Fire filtering failed"
    exit 1
fi
echo

# ============================================================
# Step 2: Create sliding windows from filtered data
# ============================================================
echo "[2/6] Creating sliding windows..."
python3 tilling_src/02_sliding_windows_filtered.py
if [ $? -ne 0 ]; then
    echo "ERROR: Sliding window creation failed"
    exit 1
fi
echo

# ============================================================
# Step 3: Spatial tiling
# ============================================================
echo "[3/6] Creating spatial regions..."
python3 tilling_src/01_spatial_tiling.py "$@"
if [ $? -ne 0 ]; then
    echo "ERROR: Spatial tiling failed"
    exit 1
fi
echo

# ============================================================
# Step 4: Temporal segmentation
# ============================================================
echo "[4/6] Creating temporal segments..."
python3 tilling_src/02_temporal_segmentation.py
if [ $? -ne 0 ]; then
    echo "ERROR: Temporal segmentation failed"
    exit 1
fi
echo

# ============================================================
# Step 5: Environment assembly
# ============================================================
echo "[5/6] Assembling environments..."
python3 tilling_src/03_environment_assembly.py
if [ $? -ne 0 ]; then
    echo "ERROR: Environment assembly failed"
    exit 1
fi
echo

# ============================================================
# Step 6: Create dataset splits
# ============================================================
echo "[6/6] Creating train/val/test splits..."
python3 tilling_src/04_dataset_split.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset splitting failed"
    exit 1
fi
echo

echo "============================================================"
echo " PIPELINE COMPLETE!"
echo "============================================================"
echo
echo "Data location: $REPO_ROOT/tilling_data/environments/"
echo "Splits created: train_split.json, val_split.json, test_split.json"
echo
echo "Next: Run training with the quality data"
echo

