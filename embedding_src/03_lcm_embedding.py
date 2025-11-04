"""
03 - Land Cover Map (LCM) Embedding
Processes Land Cover Map shapefile data and rasterizes to 400m grid
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pathlib import Path
import json

output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("LAND COVER MAP (LCM) EMBEDDING")
print("=" * 80)

# ============================================================================
# 1. LOAD GRID CONFIGURATION
# ============================================================================
print("\n[1/6] Loading grid configuration...")

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
# 2. LOAD LAND COVER MAP SHAPEFILES
# ============================================================================
print("\n[2/6] Loading Land Cover Map shapefiles...")

lcm_dir = Path('../data/LandCoverMap')
shp_files = list(lcm_dir.rglob('*.shp'))

print(f"Found {len(shp_files)} shapefiles")
for f in shp_files[:5]:
    print(f"  - {f.parent.name}/{f.name}")

# Load and combine all shapefiles
gdfs = []

for shp_file in shp_files:
    try:
        gdf = gpd.read_file(shp_file)
        # Reproject to target CRS if needed
        if gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdfs.append(gdf)
        print(f"  Loaded: {shp_file.name} ({len(gdf)} features)")
    except Exception as e:
        print(f"  Error loading {shp_file.name}: {e}")

if gdfs:
    gdf_lcm = pd.concat(gdfs, ignore_index=True)
    print(f"\nTotal features: {len(gdf_lcm):,}")
    print(f"CRS: {gdf_lcm.crs}")
else:
    print("ERROR: No shapefiles loaded successfully.")
    exit(1)

# ============================================================================
# 3. IDENTIFY LAND COVER CLASSES
# ============================================================================
print("\n[3/6] Identifying land cover classes...")

print(f"Available columns: {gdf_lcm.columns.tolist()}")

# Find the column containing land cover classes
class_columns = [col for col in gdf_lcm.columns
                 if any(keyword in col.upper() for keyword in ['CLASS', 'CODE', 'LCOV', 'TYPE'])]

print(f"Potential class columns: {class_columns}")

if class_columns:
    class_col = class_columns[0]
else:
    # Fallback: use the first non-geometry column
    class_col = [col for col in gdf_lcm.columns if col != 'geometry'][0]

print(f"Using class column: '{class_col}'")
print(f"\nUnique classes: {gdf_lcm[class_col].nunique()}")
print(gdf_lcm[class_col].value_counts().head(10))

# ============================================================================
# 4. CREATE INTEGER ENCODING
# ============================================================================
print("\n[4/6] Creating integer encoding for classes...")

unique_classes = sorted(gdf_lcm[class_col].dropna().unique())
class_to_id = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}  # Start from 1, 0 for no data
class_to_id[np.nan] = 0
id_to_class = {idx: cls for cls, idx in class_to_id.items()}

print(f"Number of land cover classes: {len(unique_classes)}")
print(f"\nClass mapping (first 10):")
for cls, idx in list(class_to_id.items())[:10]:
    print(f"  {cls} -> {idx}")

# Add integer ID to GeoDataFrame
gdf_lcm['class_id'] = gdf_lcm[class_col].map(class_to_id)
gdf_lcm['class_id'] = gdf_lcm['class_id'].fillna(0).astype(int)

# ============================================================================
# 5. RASTERIZE TO GRID
# ============================================================================
print("\n[5/6] Rasterizing to 400m grid...")

# Prepare shapes for rasterization
shapes = [(geom, value) for geom, value in zip(gdf_lcm.geometry, gdf_lcm.class_id)
          if geom is not None and geom.is_valid]

print(f"Rasterizing {len(shapes):,} features to {width}x{height} grid...")
print("This may take several minutes...")

# Rasterize
lcm_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=target_transform,
    fill=0,  # Background value for areas without data
    dtype='uint16',
    all_touched=True  # Include pixels touched by polygons
)

print(f"Rasterization completed")
print(f"Unique values: {len(np.unique(lcm_raster))}")
print(f"Value range: [{lcm_raster.min()}, {lcm_raster.max()}]")

# ============================================================================
# 6. SAVE EMBEDDED DATA
# ============================================================================
print("\n[6/6] Saving embedded data...")

# Save raster as GeoTIFF
output_path = output_dir / 'lcm_embedded.tif'

output_meta = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'width': width,
    'height': height,
    'count': 1,
    'crs': target_crs,
    'transform': target_transform,
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256
}

with rasterio.open(output_path, 'w', **output_meta) as dst:
    dst.write(lcm_raster, 1)
    dst.set_band_description(1, 'lcm_class_id')

print(f"Saved embedded data to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save class mapping
class_mapping = {
    'num_classes': len(unique_classes) + 1,  # +1 for no-data class
    'class_to_id': {str(k): int(v) for k, v in class_to_id.items() if pd.notna(k)},
    'id_to_class': {int(k): str(v) for k, v in id_to_class.items() if pd.notna(v)},
    'class_counts': {int(u): int(c) for u, c in zip(*np.unique(lcm_raster, return_counts=True))}
}

mapping_path = output_dir / 'lcm_class_mapping.json'
with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, indent=2, ensure_ascii=False)

print(f"Saved class mapping to: {mapping_path}")

print("\n" + "=" * 80)
print("LCM EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Processed {len(gdf_lcm):,} features")
print(f"✓ {len(unique_classes)} land cover classes")
print(f"✓ Rasterized to {width}x{height} grid")
print(f"✓ Ready for next stage")

