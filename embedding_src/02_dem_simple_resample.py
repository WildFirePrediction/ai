"""
DEM Embedding - Simple Approach
Just resample DEM to 400m in its native CRS using rasterio
No manual pixel iteration - let rasterio handle it
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import json

output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("DEM EMBEDDING - SIMPLE RESAMPLE (NATIVE CRS)")
print("=" * 80)

# Load DEM
dem_path = Path('../data/DigitalElevationModel/90m_GRS80.tif')

with rasterio.open(dem_path) as src:
    # Calculate new dimensions for 400m resolution
    scale_factor = 90 / 400  # 0.225
    
    new_width = int(src.width * scale_factor)
    new_height = int(src.height * scale_factor)
    
    print(f"Source: {src.width} x {src.height} @ 90m")
    print(f"Target: {new_width} x {new_height} @ 400m")
    
    # Read and resample
    dem_data = src.read(
        1,
        out_shape=(new_height, new_width),
        resampling=Resampling.bilinear
    )
    
    # Get resampled transform
    transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height)
    )
    
    nodata = src.nodata if src.nodata is not None else -9999
    crs = src.crs
    
    print(f"\nData:")
    print(f"  CRS: {crs}")
    print(f"  NoData: {nodata}")
    
    # Clean nodata
    dem_data[dem_data == nodata] = 0
    
    valid_pct = (dem_data > 0).sum() / dem_data.size * 100
    print(f"  Valid: {(dem_data > 0).sum():,} ({valid_pct:.2f}%)")
    print(f"  Range: [{dem_data[dem_data > 0].min():.2f}, {dem_data[dem_data > 0].max():.2f}]m")

# Calculate slope and aspect
print("\nCalculating slope/aspect...")
dy, dx = np.gradient(dem_data, 400)

slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad)

aspect_rad = np.arctan2(-dy, dx)
aspect_deg = (90 - np.degrees(aspect_rad)) % 360

mask = dem_data == 0
slope_deg[mask] = 0
aspect_deg[mask] = 0

# Normalize
print("Normalizing...")
valid_dem = dem_data > 0
if valid_dem.any():
    dem_min = dem_data[valid_dem].min()
    dem_max = dem_data[valid_dem].max()
    dem_norm = np.zeros_like(dem_data)
    dem_norm[valid_dem] = (dem_data[valid_dem] - dem_min) / (dem_max - dem_min + 1e-8)
else:
    dem_norm = dem_data
    dem_min, dem_max = 0, 1

slope_norm = np.clip(slope_deg / 90.0, 0, 1)
aspect_norm = aspect_deg / 360.0

# Save
print("Saving...")
stacked = np.stack([dem_norm, slope_norm, aspect_norm], axis=0).astype(np.float32)

output_path = output_dir / 'dem_slope_aspect_native.tif'

with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=new_height,
    width=new_width,
    count=3,
    dtype=np.float32,
    crs=crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(stacked)

print(f"Saved: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Metadata
metadata = {
    'crs': crs.to_string(),
    'grid_size': [new_width, new_height],
    'resolution': 400,
    'coverage_percent': float(valid_pct),
    'dem_range': [float(dem_min), float(dem_max)]
}

with open(output_dir / 'dem_native_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 80)
print(f"✓ Complete! Coverage: {valid_pct:.1f}%")
print("=" * 80)
