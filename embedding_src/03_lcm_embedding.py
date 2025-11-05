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
from tqdm import tqdm
import json
import time

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
print("Note: Processing in batches to conserve memory...")

# ============================================================================
# 3. IDENTIFY LAND COVER CLASSES - SAMPLE FIRST
# ============================================================================
print("\n[3/6] Identifying land cover classes from sample...")

# Load a few files to identify columns and classes
sample_gdfs = []
for shp_file in shp_files[:min(10, len(shp_files))]:
    try:
        gdf = gpd.read_file(shp_file)
        if gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        sample_gdfs.append(gdf)
    except Exception as e:
        pass

if not sample_gdfs:
    print("ERROR: Could not load any shapefiles.")
    exit(1)

gdf_sample = pd.concat(sample_gdfs, ignore_index=True)

print(f"Available columns: {gdf_sample.columns.tolist()}")

# Find the column containing land cover classes
class_columns = [col for col in gdf_sample.columns
                 if any(keyword in col.upper() for keyword in ['CLASS', 'CODE', 'LCOV', 'TYPE'])]

print(f"Potential class columns: {class_columns}")

if class_columns:
    class_col = class_columns[0]
else:
    # Fallback: use the first non-geometry column
    class_col = [col for col in gdf_sample.columns if col != 'geometry'][0]

print(f"Using class column: '{class_col}'")

# Collect all unique classes from ALL files (memory efficient)
print("\nScanning all files for unique classes...")
print("This may take a while - please be patient...")
unique_classes_set = set()
scan_count = 0
failed_count = 0
for shp_file in tqdm(shp_files, desc="Scanning classes"):
    try:
        gdf = gpd.read_file(shp_file, columns=[class_col, 'geometry'])
        unique_classes_set.update(gdf[class_col].dropna().unique())
        scan_count += 1
        # Free memory every 1000 files
        if scan_count % 1000 == 0:
            import gc
            gc.collect()
    except Exception as e:
        failed_count += 1
        if failed_count <= 5:  # Only print first 5 errors
            print(f"\n  Warning: Failed to read {shp_file.name}: {str(e)[:50]}")

print(f"\n✓ Scanning complete!")
print(f"  Files processed: {scan_count}/{len(shp_files)}")
print(f"  Files failed: {failed_count}")
print(f"  Unique class values: {len(unique_classes_set)}")

print(f"\nSorting {len(unique_classes_set)} unique classes...")
unique_classes = sorted(list(unique_classes_set))
print(f"✓ Sorting complete!")
del unique_classes_set  # Free memory
import gc
gc.collect()

# ============================================================================
# 4. CREATE INTEGER ENCODING
# ============================================================================
print("\n[4/6] Creating integer encoding for classes...")

class_to_id = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}  # Start from 1, 0 for no data
class_to_id[np.nan] = 0
id_to_class = {idx: cls for cls, idx in class_to_id.items()}

print(f"Number of land cover classes: {len(unique_classes)}")
print(f"\nClass mapping (first 10):")
for cls, idx in list(class_to_id.items())[:10]:
    print(f"  {cls} -> {idx}")


# ============================================================================
# 5. RASTERIZE TO GRID (BATCH PROCESSING)
# ============================================================================
print("\n[5/6] Rasterizing to 400m grid (batch processing to conserve memory)...")
print("This will take a while - processing ~17k shapefiles...")

# Initialize empty raster
lcm_raster = np.zeros((height, width), dtype='uint16')

# Process files in batches
batch_size = 100  # Process 100 files at a time
n_batches = (len(shp_files) + batch_size - 1) // batch_size

print(f"Processing {len(shp_files)} files in {n_batches} batches of {batch_size}...")
print(f"Estimated time: {n_batches * 2}+ seconds (~{n_batches * 2 / 60:.1f} minutes)")

start_time = time.time()

for batch_idx in tqdm(range(n_batches), desc="Rasterizing batches", unit="batch"):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(shp_files))
    batch_files = shp_files[start_idx:end_idx]

    # Load batch
    batch_gdfs = []
    batch_errors = 0
    for shp_file in batch_files:
        try:
            gdf = gpd.read_file(shp_file, columns=[class_col, 'geometry'])
            if gdf.crs and gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            gdf['class_id'] = gdf[class_col].map(class_to_id).fillna(0).astype(int)
            batch_gdfs.append(gdf)
        except Exception as e:
            batch_errors += 1

    if not batch_gdfs:
        continue

    # Combine batch
    gdf_batch = pd.concat(batch_gdfs, ignore_index=True)

    # Prepare shapes for rasterization
    shapes = [(geom, value) for geom, value in zip(gdf_batch.geometry, gdf_batch.class_id)
              if geom is not None and geom.is_valid]

    # Rasterize batch (merge=add will overlay on existing raster)
    batch_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=target_transform,
        fill=0,
        dtype='uint16',
        all_touched=True
    )

    # Merge with main raster (later values overwrite earlier ones)
    lcm_raster = np.where(batch_raster > 0, batch_raster, lcm_raster)

    # Free memory explicitly
    del gdf_batch, batch_gdfs, shapes, batch_raster
    import gc
    gc.collect()

    # Print progress every 20 batches
    if (batch_idx + 1) % 20 == 0:
        elapsed = time.time() - start_time
        rate = (batch_idx + 1) / elapsed
        remaining = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
        print(f"    Progress: {batch_idx + 1}/{n_batches} batches | Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min | Mem: {lcm_raster.nbytes / 1024 / 1024:.1f}MB")

elapsed_total = time.time() - start_time
print(f"\nBatch processing completed in {elapsed_total/60:.1f} minutes")

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
print(f"✓ Processed {len(shp_files):,} shapefiles")
print(f"✓ {len(unique_classes)} land cover classes")
print(f"✓ Rasterized to {width}x{height} grid")
print(f"✓ Ready for next stage")

