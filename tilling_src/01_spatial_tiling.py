"""
01 - Spatial Episode Region Extraction
Creates per-episode spatial regions where each "tile" is a single 400m grid cell.
For each fire episode we extract the bounding box (in grid coordinates) of all detections
and expand it by PAD=5 cells in all directions (if possible) so the agent has access to
full fire extent + margin.

Output:
  tilled_data/episode_regions.parquet  (index of regions)
  tilled_data/regions/episode_region_<episode_id>.npz  (static cropped arrays)

Use --max-episodes N for a quick smoke test.
"""

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).parent.parent / 'src').resolve()))

import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm

script_dir = _Path(__file__).parent
root_dir = script_dir.parent
embedded_dir = root_dir / 'embedded_data'
tilling_dir = root_dir / 'tilling_data'
regions_dir = tilling_dir / 'regions'
regions_dir.mkdir(parents=True, exist_ok=True)

PAD = 5  # cells of padding around episode extent
RESOLUTION = 400  # meters per pixel/cell (one cell = one tile in new spec)

print("=" * 80)
print("EPISODE REGION EXTRACTION")
print("=" * 80)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--max-episodes', type=int, default=None, help='Limit number of episodes processed (for testing)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. LOAD EMBEDDED GRID + FIRE DATA
# ---------------------------------------------------------------------------
print("\n[1/5] Loading embedded data & metadata...")
with open(embedded_dir / 'grid_metadata.json', 'r') as f:
    grid_meta = json.load(f)

transform = grid_meta['transform']  # [a, b, c, d, e, f]
a, b, c, d, e, f = transform
# World coords from grid indices: x = a * col + c ; y = e * row + f
# Row index from world Y: row = (f - y)/|e| because e is negative.

state = np.load(embedded_dir / 'state_vectors.npz', allow_pickle=True)
continuous = state['continuous_features']  # (C, H, W)
feature_names = state['feature_names'].tolist()
H, W = continuous.shape[1], continuous.shape[2]
print(f"  Grid size: {W} x {H} cells (400m resolution)")

lcm = state.get('lcm_classes')
fsm = state.get('fsm_classes')

# Load fire detections (reclustered with weather for richest features)
df_fire = pd.read_parquet(embedded_dir / 'nasa_viirs_with_weather_reclustered.parquet')
df_episodes = pd.read_parquet(embedded_dir / 'episode_index_reclustered.parquet')
print(f"  Episodes available: {len(df_episodes)}")
print(f"  Fire detections: {len(df_fire):,}")

if args.max_episodes:
    df_episodes = df_episodes.head(args.max_episodes)
    print(f"  Limiting to first {len(df_episodes)} episodes for test run")

# ---------------------------------------------------------------------------
# Helper conversions
# ---------------------------------------------------------------------------
def world_to_indices(x: float, y: float):
    """Convert world (EPSG:5179) meters to (row, col)."""
    col = int(round((x - c) / a))
    row = int(round((f - y) / abs(e)))  # e negative
    return row, col

def indices_to_world(row: int, col: int):
    x = a * col + c
    y = f - abs(e) * row
    return x, y

# ---------------------------------------------------------------------------
# 2. BUILD PER-EPISODE REGION DEFINITIONS
# ---------------------------------------------------------------------------
print("\n[2/5] Computing episode regions (extent + padding)...")
regions = []

for _, ep in tqdm(df_episodes.iterrows(), total=len(df_episodes), desc='  Episodes'):
    ep_id = int(ep['episode_id'])
    x_min, x_max = ep['x_min'], ep['x_max']
    y_min, y_max = ep['y_min'], ep['y_max']

    # Convert world bounds to grid indices
    r_min, c_min = world_to_indices(x_min, y_max)  # y_max is northern edge (smaller row index)
    r_max, c_max = world_to_indices(x_max, y_min)

    # Normalize ordering
    r0 = max(0, min(r_min, r_max) - PAD)
    r1 = min(H, max(r_min, r_max) + PAD + 1)  # exclusive end
    c0 = max(0, min(c_min, c_max) - PAD)
    c1 = min(W, max(c_min, c_max) + PAD + 1)

    height = r1 - r0
    width = c1 - c0

    # Skip invalid (empty) regions
    if height <= 0 or width <= 0:
        continue

    # Episode detection subset (for stats)
    ep_fire = df_fire[df_fire['episode_id'] == ep_id]
    n_det = len(ep_fire)

    regions.append({
        'episode_id': ep_id,
        'row_start': r0,
        'row_end': r1,
        'col_start': c0,
        'col_end': c1,
        'height': height,
        'width': width,
        'n_detections': n_det,
        'duration_hours': float(ep['duration_hours']),
        'spatial_extent_km': float(ep['spatial_extent_km']),
        'time_start': ep['time_start'],
        'time_end': ep['time_end']
    })

print(f"  Regions computed: {len(regions)}")

# ---------------------------------------------------------------------------
# 3. EXTRACT & SAVE STATIC CROPS
# ---------------------------------------------------------------------------
print("\n[3/5] Saving static cropped arrays per episode...")
region_records = []

for reg in tqdm(regions, desc='  Saving regions'):
    r0, r1 = reg['row_start'], reg['row_end']
    c0, c1 = reg['col_start'], reg['col_end']

    continuous_crop = continuous[:, r0:r1, c0:c1].astype(np.float32)
    save_dict = {
        'episode_id': reg['episode_id'],
        'continuous_features': continuous_crop,
        'feature_names': np.array(feature_names, dtype=object),
        'grid_coords': {
            'row_start': r0, 'row_end': r1, 'col_start': c0, 'col_end': c1
        },
        'world_bounds': {
            'x_min': indices_to_world(r1 - 1, c0)[0],  # south-west corner
            'x_max': indices_to_world(r0, c1 - 1)[0],  # north-east corner x
            'y_min': indices_to_world(r1 - 1, c0)[1],  # south-west y
            'y_max': indices_to_world(r0, c1 - 1)[1]   # north-east y
        }
    }
    if lcm is not None:
        save_dict['lcm_classes'] = lcm[r0:r1, c0:c1].astype(np.uint16)
    if fsm is not None:
        save_dict['fsm_classes'] = fsm[r0:r1, c0:c1].astype(np.uint16)

    np.savez_compressed(regions_dir / f'episode_region_{reg["episode_id"]:05d}.npz', **save_dict)

    region_records.append(reg)

# ---------------------------------------------------------------------------
# 4. SAVE INDEX PARQUET
# ---------------------------------------------------------------------------
print("\n[4/5] Writing region index parquet...")
df_regions = pd.DataFrame(region_records)
index_path = tilling_dir / 'episode_regions.parquet'
df_regions.to_parquet(index_path, index=False)
print(f"  Saved: {index_path}")

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------
print("\n[5/5] Summary")
print(f"  Episodes processed: {len(df_regions)}")
print(f"  Mean region size (cells): {df_regions['height'].mean():.1f} x {df_regions['width'].mean():.1f}")
print(f"  Mean detections per episode: {df_regions['n_detections'].mean():.1f}")
print(f"  Mean duration hours: {df_regions['duration_hours'].mean():.1f}")

summary = {
    'episodes_processed': int(len(df_regions)),
    'mean_height': float(df_regions['height'].mean()),
    'mean_width': float(df_regions['width'].mean()),
    'mean_detections': float(df_regions['n_detections'].mean()),
    'mean_duration_hours': float(df_regions['duration_hours'].mean()),
    'padding_cells': PAD,
    'resolution_m': RESOLUTION,
    'feature_names': feature_names
}
with open(tilling_dir / 'episode_region_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved summary: {tilling_dir / 'episode_region_summary.json'}")

print("\n" + "=" * 80)
print("EPISODE REGION EXTRACTION COMPLETE")
print("=" * 80)
