"""
05 - NDVI Embedding
Processes NDVI (Normalized Difference Vegetation Index) raster data
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from pathlib import Path
from datetime import datetime
import json

# Check if xarray is available for NetCDF files
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("WARNING: xarray not installed. NetCDF files will be skipped.")

output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("NDVI EMBEDDING")
print("=" * 80)

# ============================================================================
# 1. LOAD GRID CONFIGURATION
# ============================================================================
print("\n[1/5] Loading grid configuration...")

stats_path = output_dir / 'dem_rsp_norm_stats.json'

if stats_path.exists():
    with open(stats_path, 'r') as f:
        grid_config = json.load(f)

    x0 = grid_config['grid_origin_x']
    y0 = grid_config['grid_origin_y']
    width = grid_config['grid_width']
    height = grid_config['grid_height']
    tile_size = grid_config['tile_size']

    print(f"Grid configuration loaded")
    print(f"  Origin: ({x0:.2f}, {y0:.2f})")
    print(f"  Dimensions: {width} x {height}")
else:
    print("ERROR: Grid configuration not found. Run 02_dem_rsp_embedding.py first.")
    exit(1)

target_crs = 'EPSG:5179'
y1 = y0 + height * tile_size
target_transform = from_origin(x0, y1, tile_size, tile_size)

# ============================================================================
# 2. LOAD NDVI DATA
# ============================================================================
print("\n[2/5] Loading NDVI data...")

ndvi_dir = Path('../data/NDVI')
ndvi_files = list(ndvi_dir.glob('*.tif')) + list(ndvi_dir.glob('*.img')) + list(ndvi_dir.glob('*.nc'))

print(f"Found {len(ndvi_files)} NDVI files")
for f in sorted(ndvi_files)[:10]:
    print(f"  - {f.name}")

if not ndvi_files:
    print("ERROR: No NDVI files found in '../data/NDVI'")
    print("Expected file formats: .tif, .img, .nc")
    print("Please ensure NDVI data is available before running this embedding.")
    exit(1)

# Load NDVI files and organize by date if temporal information is available
# Limit to 3 most recent files to conserve memory (64GB system)
print("Note: Processing only 3 most recent NDVI files to conserve memory...")
ndvi_data_list = []

for ndvi_file in sorted(ndvi_files)[-3:]:  # Use last 3 files (most recent)
    try:
        print(f"  Loading: {ndvi_file.name}...")

        # Handle NetCDF files
        if ndvi_file.suffix == '.nc':
            if not HAS_XARRAY:
                print(f"    Skipping NetCDF file (xarray not installed)")
                continue

            # Load NetCDF with xarray
            ds = xr.open_dataset(ndvi_file)

            # Find NDVI variable (common names: NDVI, _250m_16_days_NDVI, etc.)
            ndvi_var = None
            for var in ds.data_vars:
                if 'NDVI' in str(var).upper() or '250m_16_days_NDVI' in str(var):
                    ndvi_var = var
                    break

            if ndvi_var is None:
                print(f"    Error: Could not find NDVI variable in {ndvi_file.name}")
                print(f"    Available variables: {list(ds.data_vars)}")
                continue

            print(f"    Using variable: {ndvi_var}")

            # Extract NDVI data (take mean across time if temporal dimension exists)
            ndvi_data_raw = ds[ndvi_var]

            # Handle time dimension
            if 'time' in ndvi_data_raw.dims:
                print(f"    Found time dimension with {len(ndvi_data_raw.time)} timesteps")
                # Take temporal mean to reduce to 2D
                ndvi_data = ndvi_data_raw.mean(dim='time').values
            else:
                ndvi_data = ndvi_data_raw.values

            # Get spatial info from NetCDF
            if 'lat' in ds and 'lon' in ds:
                lats = ds['lat'].values
                lons = ds['lon'].values

                # Create affine transform from lat/lon
                from rasterio.transform import from_bounds
                ndvi_transform = from_bounds(
                    lons.min(), lats.min(), lons.max(), lats.max(),
                    ndvi_data.shape[1], ndvi_data.shape[0]
                )
                ndvi_crs = 'EPSG:4326'  # WGS84 for lat/lon
            else:
                # Fallback: assume some default transform
                print(f"    Warning: No spatial coordinates found, using default")
                ndvi_transform = from_origin(0, 0, 250, 250)  # MODIS 250m resolution
                ndvi_crs = 'EPSG:4326'

            ds.close()

            # MODIS NDVI scale factor (typically scaled by 10000)
            if ndvi_data.max() > 100:  # Data is likely scaled
                ndvi_data = ndvi_data / 10000.0
                print(f"    Applied MODIS scale factor (/ 10000)")

        else:
            # Handle rasterio-compatible files (GeoTIFF, IMG, etc.)
            with rasterio.open(ndvi_file) as src:
                ndvi_data = src.read(1)
                ndvi_transform = src.transform
                ndvi_crs = src.crs

        # Try to extract date from filename
        filename = ndvi_file.stem
        date_str = None
        # Add date extraction logic if your files have dates in names

        ndvi_data_list.append({
            'file': ndvi_file,
            'data': ndvi_data,
            'transform': ndvi_transform,
            'crs': ndvi_crs,
            'date': date_str
        })
        print(f"    Loaded: {ndvi_file.name} - Shape: {ndvi_data.shape}, Range: [{np.nanmin(ndvi_data):.3f}, {np.nanmax(ndvi_data):.3f}]")
    except Exception as e:
        print(f"  Error loading {ndvi_file.name}: {e}")
        import traceback
        traceback.print_exc()

ndvi_files_loaded = ndvi_data_list

print(f"\nLoaded {len(ndvi_files_loaded)} NDVI files successfully")

# ============================================================================
# 3. NORMALIZE NDVI
# ============================================================================
print("\n[3/5] Normalizing NDVI...")

def normalize_ndvi(ndvi_data):
    """Normalize NDVI from [-1, 1] to [0, 1]"""
    # Handle invalid values
    ndvi_data = np.where(np.isnan(ndvi_data), -1, ndvi_data)
    ndvi_data = np.clip(ndvi_data, -1, 1)

    # Normalize to [0, 1]
    ndvi_norm = (ndvi_data + 1) / 2.0

    return ndvi_norm

# Normalize all NDVI data
for item in ndvi_files_loaded:
    item['data_norm'] = normalize_ndvi(item['data'])

print(f"NDVI normalization completed")
print(f"Sample normalized NDVI range: [{ndvi_files_loaded[0]['data_norm'].min():.4f}, {ndvi_files_loaded[0]['data_norm'].max():.4f}]")

# ============================================================================
# 4. REPROJECT TO TARGET GRID
# ============================================================================
print("\n[4/5] Reprojecting to 400m grid...")

# Reproject all NDVI data to target grid
for i, item in enumerate(ndvi_files_loaded):
    print(f"  Reprojecting {item['file'].name}...")

    ndvi_aligned = np.zeros((height, width), dtype=np.float32)

    reproject(
        source=item['data_norm'],
        destination=ndvi_aligned,
        src_transform=item['transform'],
        src_crs=item['crs'],
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    item['data_aligned'] = ndvi_aligned

    # Free original data to save memory
    del item['data']
    del item['data_norm']

# Force garbage collection
import gc
gc.collect()

# Use the most recent NDVI data (or create temporal average)
ndvi_current = ndvi_files_loaded[0]['data_aligned']

# If multiple files, create temporal average
if len(ndvi_files_loaded) > 1:
    ndvi_stack = np.stack([item['data_aligned'] for item in ndvi_files_loaded])
    ndvi_mean = np.mean(ndvi_stack, axis=0)
    ndvi_std = np.std(ndvi_stack, axis=0)

    print(f"\nTemporal statistics:")
    print(f"  Number of time periods: {len(ndvi_files_loaded)}")
    print(f"  Mean NDVI range: [{ndvi_mean.min():.4f}, {ndvi_mean.max():.4f}]")
    print(f"  Temporal std range: [{ndvi_std.min():.4f}, {ndvi_std.max():.4f}]")

    # Use mean as current NDVI
    ndvi_current = ndvi_mean

    # Free stack
    del ndvi_stack
    gc.collect()

print(f"\nCurrent NDVI statistics:")
print(f"  Shape: {ndvi_current.shape}")
print(f"  Range: [{ndvi_current.min():.4f}, {ndvi_current.max():.4f}]")
print(f"  Mean: {ndvi_current.mean():.4f}")

# ============================================================================
# 5. SAVE EMBEDDED DATA
# ============================================================================
print("\n[5/5] Saving embedded data...")

# Save as GeoTIFF
output_path = output_dir / 'ndvi_embedded.tif'

# Determine number of bands
if ndvi_files_loaded and len(ndvi_files_loaded) > 1:
    # Save multiple time periods as bands (max 26 for bi-weekly)
    num_bands = min(len(ndvi_files_loaded), 26)
else:
    # Save single band
    num_bands = 1

output_meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': width,
    'height': height,
    'count': num_bands,
    'crs': target_crs,
    'transform': target_transform,
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256
}

with rasterio.open(output_path, 'w', **output_meta) as dst:
    if num_bands == 1:
        dst.write(ndvi_current.astype('float32'), 1)
        dst.set_band_description(1, 'ndvi_norm')
    else:
        for i, item in enumerate(ndvi_files_loaded[:num_bands]):
            dst.write(item['data_aligned'].astype('float32'), i + 1)
            date_str = item['date'] if item['date'] else f't{i}'
            dst.set_band_description(i + 1, f'ndvi_norm_{date_str}')

print(f"Saved embedded data to: {output_path}")
print(f"Bands: {num_bands}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save NDVI statistics
ndvi_stats = {
    'num_time_periods': len(ndvi_files_loaded) if ndvi_files_loaded else 0,
    'normalization': 'range [-1, 1] to [0, 1]',
    'current_ndvi_mean': float(ndvi_current.mean()),
    'current_ndvi_std': float(ndvi_current.std()),
    'current_ndvi_min': float(ndvi_current.min()),
    'current_ndvi_max': float(ndvi_current.max())
}

if ndvi_files_loaded and len(ndvi_files_loaded) > 1 and 'ndvi_mean' in locals():
    ndvi_stats['temporal_mean'] = float(ndvi_mean.mean())
    ndvi_stats['temporal_std'] = float(ndvi_std.mean())

stats_path = output_dir / 'ndvi_norm_stats.json'
with open(stats_path, 'w') as f:
    json.dump(ndvi_stats, f, indent=2)

print(f"Saved NDVI statistics to: {stats_path}")

print("\n" + "=" * 80)
print("NDVI EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Processed {len(ndvi_files_loaded) if ndvi_files_loaded else 0} NDVI files")
print(f"✓ Normalized to [0, 1] range")
print(f"✓ Reprojected to {width}x{height} @ 400m resolution")
print(f"✓ Ready for next stage")

