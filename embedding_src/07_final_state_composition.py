"""
07 - Final State Composition
Combines all embedded data sources into final state vectors for RL training
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import json
from pathlib import Path

# Handle path regardless of where script is run from
script_dir = Path(__file__).parent
output_dir = script_dir.parent / 'embedded_data'
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("FINAL STATE COMPOSITION")
print("=" * 80)

# ============================================================================
# 1. LOAD ALL EMBEDDED DATA
# ============================================================================
print("\n[1/5] Loading all embedded data sources...")

# Load fire data (with weather)
print("\n  [Fire Data]")
nasa_path = output_dir / 'nasa_viirs_with_weather.parquet'
if nasa_path.exists():
    print(f"    Loading NASA VIIRS + Weather data...")
    df_fire = pd.read_parquet(nasa_path)
    print(f"    ✓ NASA VIIRS + Weather: {len(df_fire):,} fire detections")
    print(f"    ✓ Episodes: {df_fire['episode_id'].nunique():,}")
    print(f"    ✓ Includes temporal weather: w, d_x, d_y, rh, r")
else:
    # Fallback to old version
    nasa_path_old = output_dir / 'nasa_viirs_embedded.parquet'
    if nasa_path_old.exists():
        print(f"    Loading NASA VIIRS data (without weather)...")
        df_fire = pd.read_parquet(nasa_path_old)
        print(f"    ✓ NASA VIIRS: {len(df_fire):,} fire detections")
        print(f"    ✓ Episodes: {df_fire['episode_id'].nunique():,}")
    else:
        print(f"    ✗ NASA VIIRS data not found")
        df_fire = None

# Load DEM/RSP
print("\n  [Topography]")
dem_rsp_path = output_dir / 'dem_rsp_embedded.tif'
if dem_rsp_path.exists():
    print(f"    Loading DEM/RSP raster...")
    with rasterio.open(dem_rsp_path) as src:
        dem_norm = src.read(1)
        rsp_norm = src.read(2)
        grid_meta = {
            'width': src.width,
            'height': src.height,
            'transform': src.transform,
            'crs': src.crs
        }
    print(f"    ✓ DEM: {dem_norm.shape} ({dem_norm.nbytes / 1024 / 1024:.1f}MB)")
    print(f"    ✓ RSP: {rsp_norm.shape} ({rsp_norm.nbytes / 1024 / 1024:.1f}MB)")
    height, width = dem_norm.shape
else:
    print(f"    ✗ DEM/RSP data not found")
    dem_norm = None
    rsp_norm = None
    height, width = None, None

# Load LCM
print("\n  [Land Cover]")
lcm_path = output_dir / 'lcm_embedded.tif'
lcm_mapping_path = output_dir / 'lcm_class_mapping.json'
if lcm_path.exists():
    print(f"    Loading LCM raster...")
    with rasterio.open(lcm_path) as src:
        lcm_classes = src.read(1)
    print(f"    ✓ LCM: {lcm_classes.shape}, {len(np.unique(lcm_classes))} classes ({lcm_classes.nbytes / 1024 / 1024:.1f}MB)")

    if lcm_mapping_path.exists():
        with open(lcm_mapping_path, 'r') as f:
            lcm_mapping = json.load(f)
        print(f"    ✓ Mapping: {lcm_mapping['num_classes']} total classes")
else:
    print(f"    ✗ LCM data not found")
    lcm_classes = None
    lcm_mapping = None

# Load FSM
print("\n  [Forest Stand]")
fsm_path = output_dir / 'fsm_embedded.tif'
fsm_mapping_path = output_dir / 'fsm_class_mapping.json'
if fsm_path.exists():
    print(f"    Loading FSM raster...")
    with rasterio.open(fsm_path) as src:
        fsm_classes = src.read(1)
    print(f"    ✓ FSM: {fsm_classes.shape}, {len(np.unique(fsm_classes))} types ({fsm_classes.nbytes / 1024 / 1024:.1f}MB)")

    if fsm_mapping_path.exists():
        with open(fsm_mapping_path, 'r') as f:
            fsm_mapping = json.load(f)
        print(f"    ✓ Mapping: {fsm_mapping['num_types']} total types")
else:
    print(f"    ✗ FSM data not found")
    fsm_classes = None
    fsm_mapping = None

# Load NDVI
print("\n  [Vegetation]")
ndvi_path = output_dir / 'ndvi_embedded.tif'
if ndvi_path.exists():
    print(f"    Loading NDVI raster...")
    with rasterio.open(ndvi_path) as src:
        ndvi_norm = src.read(1)  # Use first band (most recent)
        num_ndvi_bands = src.count
    print(f"    ✓ NDVI: {ndvi_norm.shape}, {num_ndvi_bands} band(s) ({ndvi_norm.nbytes / 1024 / 1024:.1f}MB)")
else:
    print(f"    ✗ NDVI data not found")
    ndvi_norm = None

# Weather data (now per-detection, not a static grid)
print("\n  [Weather]")
print(f"    ℹ️  Weather is temporal (per fire detection)")
print(f"    ℹ️  Loaded with fire data in nasa_viirs_with_weather.parquet")
print(f"    ℹ️  Not creating static weather grid")
w_norm = None  # Weather handled per-detection now

# ============================================================================
# 2. VERIFY DATA ALIGNMENT
# ============================================================================
print("\n[2/5] Verifying data alignment...")

spatial_data = {
    'DEM': dem_norm,
    'RSP': rsp_norm,
    'LCM': lcm_classes,
    'FSM': fsm_classes,
    'NDVI': ndvi_norm,
    'Weather': w_norm
}

shapes = {}
for name, data in spatial_data.items():
    if data is not None:
        shapes[name] = data.shape
        print(f"  {name}: {data.shape}")

unique_shapes = set(shapes.values())
if len(unique_shapes) == 1:
    print(f"\n  ✓ All data aligned to grid: {list(unique_shapes)[0]}")
    if height is None:
        height, width = list(unique_shapes)[0]
elif len(unique_shapes) == 0:
    print(f"\n  ✗ No data loaded!")
else:
    print(f"\n  ⚠ WARNING: Data shapes are not aligned!")
    print(f"     Unique shapes: {unique_shapes}")

# ============================================================================
# 3. CREATE BASE STATE GRID
# ============================================================================
print("\n[3/5] Creating base state grid...")

# Stack continuous features
continuous_features = []
feature_names = []

if dem_norm is not None:
    continuous_features.append(dem_norm)
    feature_names.append('dem_norm')

if rsp_norm is not None:
    continuous_features.append(rsp_norm)
    feature_names.append('rsp_norm')

if ndvi_norm is not None:
    continuous_features.append(ndvi_norm)
    feature_names.append('ndvi_norm')

# Weather is now per-detection (not static grid), so skip here
# Weather will be added during tiling when creating temporal sequences

# Stack into (C, H, W) tensor
if continuous_features:
    print(f"  Stacking {len(continuous_features)} continuous features...")
    state_continuous = np.stack(continuous_features, axis=0).astype(np.float32)
    mem_mb = state_continuous.nbytes / 1024 / 1024
    print(f"  Continuous state tensor: {state_continuous.shape} ({mem_mb:.1f}MB)")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # Free individual feature arrays
    del continuous_features
    import gc
    gc.collect()
else:
    print("  ✗ No continuous features available")
    state_continuous = None

# Store categorical features separately
categorical_features = {}

if lcm_classes is not None:
    categorical_features['lcm'] = lcm_classes.astype(np.uint16)
    print(f"  LCM classes: {lcm_classes.shape}, unique: {len(np.unique(lcm_classes))}")

if fsm_classes is not None:
    categorical_features['fsm'] = fsm_classes.astype(np.uint16)
    print(f"  FSM classes: {fsm_classes.shape}, unique: {len(np.unique(fsm_classes))}")

print(f"\n  Categorical features: {list(categorical_features.keys())}")

# ============================================================================
# 4. CREATE FIRE EPISODE INDEX
# ============================================================================
print("\n[4/5] Creating fire episode index...")

if df_fire is not None:
    # Group by episode
    episodes = []

    for episode_id in df_fire['episode_id'].unique():
        if episode_id < 0:  # Skip noise
            continue

        group = df_fire[df_fire['episode_id'] == episode_id]

        episode_info = {
            'episode_id': int(episode_id),
            'start_time': group['datetime'].min(),
            'end_time': group['datetime'].max(),
            'duration_hours': float((group['datetime'].max() - group['datetime'].min()).total_seconds() / 3600),
            'num_detections': int(len(group)),
            'x_min': float(group['x'].min()),
            'x_max': float(group['x'].max()),
            'y_min': float(group['y'].min()),
            'y_max': float(group['y'].max()),
            'mean_intensity': float(group['i'].mean()),
            'max_intensity': float(group['i'].max()),
            'mean_temperature': float(group['te'].mean())
        }
        episodes.append(episode_info)

    df_episodes = pd.DataFrame(episodes)

    print(f"  Created episode index: {len(df_episodes)} episodes")
    print(f"  Mean duration: {df_episodes['duration_hours'].mean():.2f} hours")
    print(f"  Mean detections per episode: {df_episodes['num_detections'].mean():.1f}")

    # Save episode index
    episode_path = output_dir / 'episode_index.parquet'
    df_episodes.to_parquet(episode_path, index=False)
    print(f"  Saved to: {episode_path}")
else:
    print("  ✗ No fire data available for episode index")
    df_episodes = None

# ============================================================================
# 5. SAVE FINAL STATE DATA
# ============================================================================
print("\n[5/5] Saving final state data...")

# Save all state components as NPZ
state_output = output_dir / 'state_vectors.npz'

save_dict = {}

if state_continuous is not None:
    save_dict['continuous_features'] = state_continuous
    save_dict['feature_names'] = np.array(feature_names, dtype=object)

# Add categorical features
for name, data in categorical_features.items():
    save_dict[f'{name}_classes'] = data

if save_dict:
    np.savez_compressed(state_output, **save_dict)

    print(f"  Saved state vectors to: {state_output}")
    print(f"  File size: {state_output.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n  Saved arrays:")
    for key in save_dict.keys():
        if key != 'feature_names':
            print(f"    {key}: {save_dict[key].shape}")
else:
    print("  ✗ No data to save!")

# Save grid metadata
if 'grid_meta' in locals() and grid_meta:
    grid_metadata = {
        'width': int(width),
        'height': int(height),
        'tile_size': 400,
        'crs': str(grid_meta['crs']),
        'transform': [float(x) for x in grid_meta['transform'][:6]],
        'num_continuous_features': len(feature_names) if feature_names else 0,
        'continuous_feature_names': feature_names,
        'categorical_features': {}
    }

    if lcm_mapping:
        grid_metadata['categorical_features']['lcm'] = {
            'num_classes': lcm_mapping['num_classes'],
            'embedding_dim_recommended': 16
        }

    if fsm_mapping:
        grid_metadata['categorical_features']['fsm'] = {
            'num_classes': fsm_mapping['num_types'],
            'embedding_dim_recommended': 16
        }

    metadata_path = output_dir / 'grid_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(grid_metadata, f, indent=2)

    print(f"  Saved grid metadata to: {metadata_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL STATE COMPOSITION COMPLETE")
print("=" * 80)

print("\n📊 Summary:")
print(f"  Grid: {width} x {height} @ 400m resolution")
print(f"  CRS: EPSG:5179")
print(f"  Continuous features: {len(feature_names) if feature_names else 0}")
print(f"  Categorical features: {len(categorical_features)}")

if df_episodes is not None:
    print(f"  Fire episodes: {len(df_episodes):,}")
    print(f"  Fire detections: {len(df_fire):,}")

print("\n💾 Output files:")
for f in sorted(output_dir.glob('*')):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")

print("\n🚀 Ready for RL training!")
print("=" * 80)

