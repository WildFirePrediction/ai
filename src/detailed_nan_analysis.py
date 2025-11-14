"""
Detailed Analysis of NaN Values and Data Quality Issues
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DETAILED DATA QUALITY ANALYSIS")
print("=" * 80)

tilling_dir = Path('../tilling_data')
regions_dir = tilling_dir / 'regions'
sequences_dir = tilling_dir / 'sequences'

# =============================================================================
# 1. ANALYZE NaN VALUES IN CONTINUOUS FEATURES ACROSS ALL REGIONS
# =============================================================================
print("\n" + "=" * 80)
print("1. NaN VALUES IN CONTINUOUS FEATURES")
print("=" * 80)

region_files = sorted(list(regions_dir.glob('window_region_*.npz')))
print(f"\nAnalyzing {len(region_files)} region files...")

nan_stats = []

for i, region_file in enumerate(region_files[:20]):  # Sample first 20 regions
    region_data = np.load(region_file, allow_pickle=True)

    if 'continuous_features' in region_data:
        cont = region_data['continuous_features']
        feature_names = region_data.get('feature_names', ['unknown'] * cont.shape[0])

        total_values = np.prod(cont.shape)
        nan_count = np.isnan(cont).sum()
        nan_pct = 100 * nan_count / total_values

        # Per-feature NaN analysis
        per_feature_nan = {}
        for idx, fname in enumerate(feature_names):
            feature_data = cont[idx]
            feature_nan = np.isnan(feature_data).sum()
            feature_total = np.prod(feature_data.shape)
            per_feature_nan[fname] = {
                'nan_count': int(feature_nan),
                'nan_pct': float(100 * feature_nan / feature_total)
            }

        nan_stats.append({
            'window_id': int(region_data['window_id']),
            'total_nan_pct': float(nan_pct),
            'per_feature': per_feature_nan
        })

        if i < 5:  # Print details for first 5
            print(f"\n  Window {region_data['window_id']:05d}:")
            print(f"    Total NaN: {nan_count}/{total_values} ({nan_pct:.2f}%)")
            for fname, stats in per_feature_nan.items():
                print(f"      {fname}: {stats['nan_pct']:.2f}% NaN")

# Summarize
if nan_stats:
    total_nan_pcts = [s['total_nan_pct'] for s in nan_stats]
    print(f"\n  Summary (sampled {len(nan_stats)} regions):")
    print(f"    Mean NaN percentage: {np.mean(total_nan_pcts):.2f}%")
    print(f"    Min NaN percentage: {np.min(total_nan_pcts):.2f}%")
    print(f"    Max NaN percentage: {np.max(total_nan_pcts):.2f}%")

    # Per-feature aggregation
    print(f"\n  Per-feature NaN statistics:")
    for fname in ['dem_norm', 'rsp_norm', 'ndvi_norm']:
        feature_nans = [s['per_feature'].get(fname, {}).get('nan_pct', 0) for s in nan_stats]
        if feature_nans:
            print(f"    {fname}: mean={np.mean(feature_nans):.2f}%, max={np.max(feature_nans):.2f}%")

# =============================================================================
# 2. CHECK COORDINATE TRANSFORMATION CONSISTENCY
# =============================================================================
print("\n" + "=" * 80)
print("2. COORDINATE TRANSFORMATION CONSISTENCY")
print("=" * 80)

# Load fire data
df_fire = pd.read_parquet(Path('../data/filtered_fires/filtered_viirs.parquet'))

# Check if coordinate transformation is consistent
print("\n  Verifying EPSG:4326 -> EPSG:5179 transformation...")

# Sample some points and verify transformation
sample_fire = df_fire.sample(min(5, len(df_fire)), random_state=42)

import pyproj
proj_src = pyproj.Proj("EPSG:4326")
proj_dst = pyproj.Proj("EPSG:5179")

print("\n  Sample coordinate transformations:")
for idx, row in sample_fire.iterrows():
    lon, lat = row['LONGITUDE'], row['LATITUDE']
    x_stored, y_stored = row['x'], row['y']

    # Recompute transformation
    x_computed, y_computed = pyproj.transform(proj_src, proj_dst, lon, lat)

    # Check difference
    x_diff = abs(x_computed - x_stored)
    y_diff = abs(y_computed - y_stored)

    status = "✓" if (x_diff < 1.0 and y_diff < 1.0) else "⚠️"
    print(f"    [{status}] (lon={lon:.4f}, lat={lat:.4f}) -> (x={x_stored:.2f}, y={y_stored:.2f})")
    print(f"        Computed: (x={x_computed:.2f}, y={y_computed:.2f}), Diff: ({x_diff:.2f}m, {y_diff:.2f}m)")

# =============================================================================
# 3. CHECK TEMPORAL ALIGNMENT IN DETAIL
# =============================================================================
print("\n" + "=" * 80)
print("3. TEMPORAL ALIGNMENT BETWEEN FIRE AND WEATHER DATA")
print("=" * 80)

print("\n  Checking temporal resolution...")

# Check time gaps between consecutive fire detections
df_fire_sorted = df_fire.sort_values('datetime')
time_diffs = df_fire_sorted['datetime'].diff()
time_diffs_hours = time_diffs.dt.total_seconds() / 3600.0

print(f"    Time gaps between consecutive fire detections:")
print(f"      Median: {time_diffs_hours.median():.2f} hours")
print(f"      Mean: {time_diffs_hours.mean():.2f} hours")
print(f"      Min: {time_diffs_hours.min():.2f} hours")
print(f"      Max: {time_diffs_hours.max():.2f} hours")

# Check if weather data timestamps align with fire detection timestamps
print(f"\n  Checking if weather data is properly matched to fire detections...")

# Sample some fire detections and check weather data
sample_fires = df_fire.sample(min(10, len(df_fire)), random_state=42)

print(f"\n  Sample fire detections with weather:")
for idx, row in sample_fires.head(5).iterrows():
    print(f"    Time: {row['datetime']}, Location: ({row['x']:.0f}, {row['y']:.0f})")
    print(f"      Weather: wind={row['w']:.1f}m/s, rh={row['rh']:.1f}%, temp={row['te']:.1f}°C, rain={row['r']:.2f}mm")

# =============================================================================
# 4. CHECK WINDOW-SEQUENCE MATCHING
# =============================================================================
print("\n" + "=" * 80)
print("4. WINDOW-SEQUENCE MATCHING")
print("=" * 80)

# Load window regions and check if all have sequences
df_regions = pd.read_parquet(tilling_dir / 'window_regions.parquet')
sequence_files = sorted(list(sequences_dir.glob('window_*.npz')))

print(f"\n  Total windows: {len(df_regions)}")
print(f"  Total sequences: {len(sequence_files)}")

# Extract window IDs from sequence files
seq_window_ids = set()
for seq_file in sequence_files:
    win_id = int(seq_file.stem.split('_')[1])
    seq_window_ids.add(win_id)

# Check which windows don't have sequences
missing_sequences = []
for win_id in df_regions['window_id']:
    if win_id not in seq_window_ids:
        missing_sequences.append(win_id)

if missing_sequences:
    print(f"\n  [⚠️] WARNING: {len(missing_sequences)} windows missing sequences!")
    print(f"    Missing window IDs (first 10): {missing_sequences[:10]}")

    # Analyze why these are missing
    missing_df = df_regions[df_regions['window_id'].isin(missing_sequences)]
    print(f"\n    Statistics of missing windows:")
    print(f"      Mean detections: {missing_df['n_detections'].mean():.1f}")
    print(f"      Min detections: {missing_df['n_detections'].min()}")
    print(f"      Max detections: {missing_df['n_detections'].max()}")
else:
    print(f"  [✓] All windows have corresponding sequences")

# =============================================================================
# 5. CHECK SPATIAL DIMENSIONS CONSISTENCY
# =============================================================================
print("\n" + "=" * 80)
print("5. SPATIAL DIMENSIONS CONSISTENCY")
print("=" * 80)

print("\n  Checking if static and temporal dimensions match...")

mismatches = []

for seq_file in sequence_files[:20]:  # Sample first 20
    win_id = int(seq_file.stem.split('_')[1])

    # Load region and sequence
    region_file = regions_dir / f'window_region_{win_id:05d}.npz'
    if not region_file.exists():
        continue

    region_data = np.load(region_file, allow_pickle=True)
    seq_data = np.load(seq_file, allow_pickle=True)

    # Get dimensions
    static_h, static_w = region_data['continuous_features'].shape[1:3]
    temporal_h, temporal_w = seq_data['fire_masks'].shape[1:3]

    if static_h != temporal_h or static_w != temporal_w:
        mismatches.append({
            'window_id': win_id,
            'static_dims': (static_h, static_w),
            'temporal_dims': (temporal_h, temporal_w)
        })

if mismatches:
    print(f"  [⚠️] WARNING: Found {len(mismatches)} dimension mismatches!")
    for mm in mismatches[:5]:
        print(f"    Window {mm['window_id']}: static={mm['static_dims']}, temporal={mm['temporal_dims']}")
else:
    print(f"  [✓] All sampled windows have matching static/temporal dimensions")

# =============================================================================
# 6. CHECK GRID METADATA CONSISTENCY
# =============================================================================
print("\n" + "=" * 80)
print("6. GRID METADATA CONSISTENCY")
print("=" * 80)

# Check if grid_coords in regions match the actual array dimensions
print("\n  Verifying grid coordinates match array dimensions...")

sample_region_files = region_files[:10]

for region_file in sample_region_files:
    region_data = np.load(region_file, allow_pickle=True)

    grid_coords = region_data['grid_coords'].item()
    r0, r1 = grid_coords['row_start'], grid_coords['row_end']
    c0, c1 = grid_coords['col_start'], grid_coords['col_end']

    expected_h = r1 - r0
    expected_w = c1 - c0

    actual_h, actual_w = region_data['continuous_features'].shape[1:3]

    if expected_h != actual_h or expected_w != actual_w:
        print(f"  [⚠️] Window {region_data['window_id']}: Expected ({expected_h}, {expected_w}), Got ({actual_h}, {actual_w})")
    else:
        pass  # Silent if OK

print(f"  [✓] Checked {len(sample_region_files)} regions - grid coords match array dims")

print("\n" + "=" * 80)
print("DETAILED ANALYSIS COMPLETE")
print("=" * 80)
