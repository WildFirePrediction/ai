"""
04 - Forest Stand Map (FSM) Embedding
Processes Forest Stand Map (임상도) shapefile data from all provinces
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
print("FOREST STAND MAP (FSM) EMBEDDING")
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
# 2. LOAD FOREST STAND MAP SHAPEFILES
# ============================================================================
print("\n[2/6] Loading Forest Stand Map shapefiles from all provinces...")

fsm_dir = Path('../data/ForestStandMap')
province_dirs = [d for d in fsm_dir.iterdir() if d.is_dir()]

print(f"Found {len(province_dirs)} province directories:")
for d in province_dirs:
    print(f"  - {d.name}")

# List all shapefiles
shp_files = list(fsm_dir.rglob('*.shp'))
print(f"\nTotal shapefiles: {len(shp_files)}")

# Load and combine shapefiles from all provinces
gdfs = []

for shp_file in shp_files:
    try:
        gdf = gpd.read_file(shp_file)
        # Reproject to target CRS if needed
        if gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdfs.append(gdf)
        print(f"  Loaded: {shp_file.parent.name}/{shp_file.name} ({len(gdf)} features)")
    except Exception as e:
        print(f"  Error loading {shp_file.name}: {e}")

if gdfs:
    gdf_fsm = pd.concat(gdfs, ignore_index=True)
    print(f"\nTotal features: {len(gdf_fsm):,}")
    print(f"CRS: {gdf_fsm.crs}")
else:
    print("ERROR: No shapefiles loaded successfully.")
    exit(1)

# ============================================================================
# 3. IDENTIFY FOREST TYPE CLASSES
# ============================================================================
print("\n[3/6] Identifying forest type classes...")

print(f"Available columns: {gdf_fsm.columns.tolist()}")

# Find columns related to forest type (임상)
forest_columns = [
    col for col in gdf_fsm.columns
    if any(keyword in col.upper() for keyword in ['임상', 'IMSNAG', 'FRST', 'TYPE', 'CLASS', 'KIND'])
]

print(f"Potential forest type columns: {forest_columns}")

if forest_columns:
    forest_col = forest_columns[0]
else:
    # Fallback: use the first non-geometry column
    forest_col = [col for col in gdf_fsm.columns if col != 'geometry'][0]

print(f"Using forest type column: '{forest_col}'")
print(f"\nUnique forest types: {gdf_fsm[forest_col].nunique()}")
print(gdf_fsm[forest_col].value_counts().head(10))

# ============================================================================
# 4. CREATE INTEGER ENCODING
# ============================================================================
print("\n[4/6] Creating integer encoding for forest types...")

unique_types = sorted(gdf_fsm[forest_col].dropna().unique())
type_to_id = {ftype: idx + 1 for idx, ftype in enumerate(unique_types)}  # Start from 1, 0 for no data
type_to_id[np.nan] = 0
id_to_type = {idx: ftype for ftype, idx in type_to_id.items()}

print(f"Number of forest types: {len(unique_types)}")
print(f"\nForest type mapping (first 15):")
for ftype, idx in list(type_to_id.items())[:15]:
    print(f"  {ftype} -> {idx}")

# Add integer ID to GeoDataFrame
gdf_fsm['type_id'] = gdf_fsm[forest_col].map(type_to_id)
gdf_fsm['type_id'] = gdf_fsm['type_id'].fillna(0).astype(int)

# ============================================================================
# 5. RASTERIZE TO GRID
# ============================================================================
print("\n[5/6] Rasterizing to 400m grid...")

# Prepare shapes for rasterization
shapes = [(geom, value) for geom, value in zip(gdf_fsm.geometry, gdf_fsm.type_id)
          if geom is not None and geom.is_valid]

print(f"Rasterizing {len(shapes):,} features to {width}x{height} grid...")
print("This may take several minutes for large datasets...")

# Rasterize
fsm_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=target_transform,
    fill=0,  # Background value for areas without data
    dtype='uint16',
    all_touched=True  # Include pixels touched by polygons
)

print(f"Rasterization completed")
print(f"Unique values: {len(np.unique(fsm_raster))}")
print(f"Value range: [{fsm_raster.min()}, {fsm_raster.max()}]")

# ============================================================================
# 6. SAVE EMBEDDED DATA
# ============================================================================
print("\n[6/6] Saving embedded data...")

# Save raster as GeoTIFF
output_path = output_dir / 'fsm_embedded.tif'

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
    dst.write(fsm_raster, 1)
    dst.set_band_description(1, 'fsm_type_id')

print(f"Saved embedded data to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save forest type mapping
type_mapping = {
    'num_types': len(unique_types) + 1,  # +1 for no-data class
    'type_to_id': {str(k): int(v) for k, v in type_to_id.items() if pd.notna(k)},
    'id_to_type': {int(k): str(v) for k, v in id_to_type.items() if pd.notna(v)},
    'type_counts': {int(u): int(c) for u, c in zip(*np.unique(fsm_raster, return_counts=True))}
}

mapping_path = output_dir / 'fsm_class_mapping.json'
with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump(type_mapping, f, indent=2, ensure_ascii=False)

print(f"Saved type mapping to: {mapping_path}")

# Print coverage statistics
unique_vals, counts = np.unique(fsm_raster, return_counts=True)
total_pixels = fsm_raster.size

print(f"\nForest Type Coverage (Top 10):")
sorted_indices = np.argsort(counts)[::-1]
for idx in sorted_indices[:10]:
    val, count = unique_vals[idx], counts[idx]
    type_name = id_to_type.get(val, 'Unknown')
    if pd.isna(type_name):
        type_name = 'No Data'
    coverage = count / total_pixels * 100
    print(f"  {val:<5} {str(type_name)[:30]:<30} {count:>10,} pixels ({coverage:>5.2f}%)")

print("\n" + "=" * 80)
print("FSM EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Processed {len(gdf_fsm):,} features from {len(province_dirs)} provinces")
print(f"✓ {len(unique_types)} forest types")
print(f"✓ Rasterized to {width}x{height} grid")
print(f"✓ Ready for next stage")

