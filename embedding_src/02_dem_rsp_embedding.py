"""
02 - DEM & RSP Embedding
Processes Digital Elevation Model and Relative Slope Position data
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds
from pathlib import Path
import json

output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("DEM & RSP EMBEDDING")
print("=" * 80)

# ============================================================================
# 1. LOAD DEM DATA
# ============================================================================
print("\n[1/5] Loading DEM data...")

dem_path = Path('../data/DigitalElevationModel/90m_GRS80.tif')

if not dem_path.exists():
    print(f"WARNING: DEM file not found at {dem_path}")
    print("Trying alternative path...")
    dem_path = Path('../data/DigitalElevationModel/90m_GRS80.img')

with rasterio.open(dem_path) as src:
    dem_data = src.read(1)
    dem_meta = src.meta.copy()
    dem_transform = src.transform
    dem_crs = src.crs
    dem_bounds = src.bounds

print(f"DEM loaded successfully")
print(f"Shape: {dem_data.shape}")
print(f"CRS: {dem_crs}")
print(f"Bounds: {dem_bounds}")
print(f"Elevation range: [{dem_data.min():.2f}, {dem_data.max():.2f}] m")

# ============================================================================
# 2. NORMALIZE DEM
# ============================================================================
print("\n[2/5] Normalizing DEM...")

# Mask out nodata values (commonly -9999 or similar)
nodata_value = dem_meta.get('nodata', -9999)
valid_mask = dem_data != nodata_value

dem_min = float(dem_data[valid_mask].min())
dem_max = float(dem_data[valid_mask].max())

# Normalize only valid data
dem_norm = np.zeros_like(dem_data, dtype=np.float32)
dem_norm[valid_mask] = (dem_data[valid_mask] - dem_min) / (dem_max - dem_min)
dem_norm[~valid_mask] = 0  # Set nodata to 0

print(f"DEM normalized to range: [{dem_norm[valid_mask].min():.4f}, {dem_norm[valid_mask].max():.4f}]")
print(f"Valid pixels: {valid_mask.sum():,} / {dem_data.size:,}")

# ============================================================================
# 3. LOAD RSP DATA
# ============================================================================
print("\n[3/5] Loading RSP data...")

rsp_dir = Path('../data/RelativeSlopePosition')

# Prefer full RSP file over province-specific files
full_rsp_path = rsp_dir / '02.전체' / 'GEOTIFF' / 'Relative Slope Position.tif'

if full_rsp_path.exists():
    rsp_path = full_rsp_path
    print(f"Found FULL RSP file: {rsp_path.name}")
else:
    # Fallback to any RSP file
    rsp_files = (list(rsp_dir.glob('*.tif')) +
                 list(rsp_dir.glob('*.img')) +
                 list(rsp_dir.glob('**/*.tif')) +
                 list(rsp_dir.glob('**/*.img')))
    
    if rsp_files:
        rsp_path = rsp_files[0]
        print(f"Found RSP file: {rsp_path.name}")
        print(f"⚠️  WARNING: Using province-specific RSP (limited coverage)")
    else:
        rsp_path = None

if rsp_path:

    with rasterio.open(rsp_path) as src:
        rsp_data = src.read(1)
        rsp_meta = src.meta.copy()
        rsp_transform = src.transform
        rsp_crs = src.crs

    print(f"RSP loaded successfully")
    print(f"Shape: {rsp_data.shape}")
    print(f"RSP range: [{rsp_data.min():.4f}, {rsp_data.max():.4f}]")
else:
    print("ERROR: No RSP files found in '../data/RelativeSlopePosition'")
    print("Expected to find RSP data files (.tif or .img format)")
    print("Please ensure RSP data is available before running this embedding.")
    exit(1)

# ============================================================================
# 4. ALIGN TO TARGET GRID (400m, EPSG:5179) - SOUTH KOREA ONLY
# ============================================================================
print("\n[4/5] Aligning to 400m grid (EPSG:5179)...")

target_crs = 'EPSG:5179'
target_resolution = 400  # meters

# SOUTH KOREA BOUNDS (32.5-39°N, 124-131.5°E)
# This clips the grid to exclude North Korea
SOUTH_KOREA_BOUNDS = {
    'x_min': 670_800,
    'y_min': 1_395_200,
    'x_max': 1_346_800,
    'y_max': 2_118_800
}

print("\n⚠️  CLIPPING TO SOUTH KOREA BOUNDS ONLY")
print(f"  X: {SOUTH_KOREA_BOUNDS['x_min']:,} to {SOUTH_KOREA_BOUNDS['x_max']:,} ({(SOUTH_KOREA_BOUNDS['x_max']-SOUTH_KOREA_BOUNDS['x_min'])/1000:.1f} km)")
print(f"  Y: {SOUTH_KOREA_BOUNDS['y_min']:,} to {SOUTH_KOREA_BOUNDS['y_max']:,} ({(SOUTH_KOREA_BOUNDS['y_max']-SOUTH_KOREA_BOUNDS['y_min'])/1000:.1f} km)")
print(f"  (Excludes North Korea - DMZ is at Y ≈ 1,450,000)")

# Calculate target transform and dimensions
# If DEM is not in EPSG:5179, we need to transform its bounds first

if str(dem_crs) != 'EPSG:5179':
    print(f"\nDEM CRS ({dem_crs}) differs from target CRS (EPSG:5179)")
    print("Calculating transformed bounds...")

    # Calculate transform from source to target CRS
    transform, width, height = calculate_default_transform(
        dem_crs, target_crs,
        dem_data.shape[1], dem_data.shape[0],
        *dem_bounds
    )

    # Get bounds in target CRS
    west, south, east, north = transform_bounds(dem_crs, target_crs, *dem_bounds)

    print(f"Full DEM bounds (EPSG:5179): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")
else:
    west, south, east, north = dem_bounds
    print(f"Full DEM bounds (EPSG:5179): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")

# Clip to South Korea bounds
west = max(west, SOUTH_KOREA_BOUNDS['x_min'])
south = max(south, SOUTH_KOREA_BOUNDS['y_min'])
east = min(east, SOUTH_KOREA_BOUNDS['x_max'])
north = min(north, SOUTH_KOREA_BOUNDS['y_max'])

print(f"Clipped bounds (South Korea): W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")

# Snap to 400m grid
x0 = np.floor(west / target_resolution) * target_resolution
y0 = np.floor(south / target_resolution) * target_resolution
x1 = np.ceil(east / target_resolution) * target_resolution
y1 = np.ceil(north / target_resolution) * target_resolution

width = int((x1 - x0) / target_resolution)
height = int((y1 - y0) / target_resolution)

target_transform = from_origin(x0, y1, target_resolution, target_resolution)

print(f"Target grid:")
print(f"  Origin: ({x0:.2f}, {y0:.2f})")
print(f"  Dimensions: {width} x {height}")
print(f"  Resolution: {target_resolution}m")

# Reproject DEM
dem_aligned = np.zeros((height, width), dtype=np.float32)
reproject(
    source=dem_norm,
    destination=dem_aligned,
    src_transform=dem_transform,
    src_crs=dem_crs,
    dst_transform=target_transform,
    dst_crs=target_crs,
    resampling=Resampling.bilinear
)

print(f"DEM reprojected: {dem_aligned.shape}, range [{dem_aligned.min():.4f}, {dem_aligned.max():.4f}]")

# Reproject RSP
rsp_aligned = np.full((height, width), np.nan, dtype=np.float32)
reproject(
    source=rsp_data,
    destination=rsp_aligned,
    src_transform=rsp_transform,
    src_crs=rsp_crs if 'rsp_crs' in locals() else dem_crs,
    dst_transform=target_transform,
    dst_crs=target_crs,
    resampling=Resampling.bilinear,
    src_nodata=-99999,
    dst_nodata=np.nan
)

# Clip RSP to valid range [0, 1]
rsp_aligned = np.clip(rsp_aligned, 0, 1)

print(f"RSP reprojected: {rsp_aligned.shape}, range [{np.nanmin(rsp_aligned):.4f}, {np.nanmax(rsp_aligned):.4f}]")
print(f"  NaN pixels: {np.isnan(rsp_aligned).sum()}")

# ============================================================================
# 5. SAVE EMBEDDED DATA
# ============================================================================
print("\n[5/5] Saving embedded data...")

output_path = output_dir / 'dem_rsp_embedded.tif'

output_meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': width,
    'height': height,
    'count': 2,
    'crs': target_crs,
    'transform': target_transform,
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256,
    'nodata': np.nan
}

with rasterio.open(output_path, 'w', **output_meta) as dst:
    dst.write(dem_aligned, 1)
    dst.write(rsp_aligned, 2)
    dst.set_band_description(1, 'dem_norm')
    dst.set_band_description(2, 'rsp_norm')

print(f"Saved embedded data to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save normalization statistics
norm_stats = {
    'dem_min': dem_min,
    'dem_max': dem_max,
    'rsp_min': float(rsp_data.min()),
    'rsp_max': float(rsp_data.max()),
    'grid_origin_x': float(x0),
    'grid_origin_y': float(y0),
    'grid_width': int(width),
    'grid_height': int(height),
    'tile_size': int(target_resolution)
}

stats_path = output_dir / 'dem_rsp_norm_stats.json'
with open(stats_path, 'w') as f:
    json.dump(norm_stats, f, indent=2)

print(f"Saved normalization stats to: {stats_path}")

print("\n" + "=" * 80)
print("DEM & RSP EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Grid: {width} x {height} @ {target_resolution}m resolution")
print(f"✓ CRS: {target_crs}")
print(f"✓ Ready for next stage")

