"""
04 - Forest Stand Map (FSM) Embedding
Processes Forest Stand Map (임상도) shapefile data from all provinces
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pathlib import Path
import json
import time

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
print("Note: Processing in batches to conserve memory...")

# ============================================================================
# 3. IDENTIFY COLUMN NAME FROM SINGLE FILE
# ============================================================================
print("\n[3/6] Identifying forest type column from sample file...")

# Load just ONE file to identify the column
forest_col = None
for shp_file in shp_files[:5]:
    try:
        gdf = gpd.read_file(shp_file)
        print(f"Available columns: {gdf.columns.tolist()}")

        # Find columns related to forest type (임상)
        forest_columns = [
            col for col in gdf.columns
            if any(keyword in col.upper() for keyword in ['임상', 'IMSNAG', 'FRST', 'TYPE', 'CLASS', 'KIND'])
        ]

        print(f"Potential forest type columns: {forest_columns}")

        if forest_columns:
            forest_col = forest_columns[0]
        else:
            # Fallback: use the first non-geometry column
            forest_col = [col for col in gdf.columns if col != 'geometry'][0]

        print(f"Using forest type column: '{forest_col}'")
        del gdf  # Free memory immediately
        break
    except Exception as e:
        continue

if not forest_col:
    print("ERROR: Could not identify forest type column.")
    exit(1)

# ============================================================================
# 4. PROCESS FILES ONE AT A TIME (MEMORY EFFICIENT)
# ============================================================================
print(f"\n[4/6] Processing {len(shp_files)} files ONE AT A TIME (memory efficient)...")
print("Building type mapping and raster incrementally...")

# Initialize empty raster
fsm_raster = np.zeros((height, width), dtype='uint16')

# Dynamic type mapping (build as we go)
type_to_id = {}
next_id = 1  # 0 reserved for no-data

start_time = time.time()
processed_count = 0
error_count = 0

# Process each file individually
for file_idx, shp_file in enumerate(shp_files):
    try:
        # Read only the columns we need
        gdf = gpd.read_file(shp_file, columns=[forest_col, 'geometry'])

        # Reproject if needed
        if gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Add new types to mapping
        unique_types = gdf[forest_col].dropna().unique()
        for ftype in unique_types:
            if ftype not in type_to_id:
                type_to_id[ftype] = next_id
                next_id += 1

        # Map to IDs
        gdf['type_id'] = gdf[forest_col].map(type_to_id).fillna(0).astype(int)

        # Prepare shapes for rasterization
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf.type_id)
                  if geom is not None and geom.is_valid]

        if shapes:
            # Rasterize this file
            file_raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=target_transform,
                fill=0,
                dtype='uint16',
                all_touched=True
            )

            # Merge with main raster (overwrite where there's data)
            fsm_raster = np.where(file_raster > 0, file_raster, fsm_raster)

            del file_raster

        # Free memory
        del gdf, shapes
        processed_count += 1

        # Progress update every 100 files
        if (file_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (file_idx + 1) / elapsed
            remaining = (len(shp_files) - file_idx - 1) / rate if rate > 0 else 0
            coverage = np.count_nonzero(fsm_raster) / fsm_raster.size * 100
            print(f"  [{file_idx + 1}/{len(shp_files)}] Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min | "
                  f"Types: {len(type_to_id)} | Coverage: {coverage:.1f}% | Errors: {error_count}")

            # Force garbage collection every 100 files
            import gc
            gc.collect()

    except Exception as e:
        error_count += 1
        if error_count <= 5:  # Only print first few errors
            print(f"  Warning: Failed to process {shp_file.name}: {str(e)[:100]}")
        continue

elapsed_total = time.time() - start_time
print(f"\nProcessing completed in {elapsed_total/60:.1f} minutes")
print(f"Successfully processed: {processed_count}/{len(shp_files)} files")
print(f"Errors: {error_count}")

# ============================================================================
# 5. CREATE FINAL TYPE MAPPING
# ============================================================================
print("\n[5/6] Creating final type mapping...")

type_to_id[np.nan] = 0
id_to_type = {idx: ftype for ftype, idx in type_to_id.items()}

print(f"Number of forest types: {len(type_to_id) - 1}")  # -1 for nan
print(f"\nForest type mapping (first 15):")
for ftype, idx in list(type_to_id.items())[:15]:
    if pd.notna(ftype):
        print(f"  {ftype} -> {idx}")

print(f"Rasterization completed")
print(f"Unique values in raster: {len(np.unique(fsm_raster))}")
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
    'num_types': len(type_to_id),
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
print(f"✓ Processed {processed_count:,} shapefiles from {len(province_dirs)} provinces")
print(f"✓ {len(type_to_id) - 1} forest types (excluding no-data)")
print(f"✓ Rasterized to {width}x{height} grid")
print(f"✓ Ready for next stage")

