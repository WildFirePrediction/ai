"""
Tile Integrity Validation
Validates legacy 256x256 spatial tiles stored in tilting_data/tiles against tile_index.parquet.
Checks:
  - Presence of NPZ file for every retained tile
  - continuous_features shape consistency
  - categorical feature shape alignment (lcm_classes, fsm_classes)
  - No NaNs or Infs in continuous features (reports counts)
  - Episode linkage: episode_ids exist in episode index parquet
  - Detection count consistency (recomputed vs stored num_detections)
  - Summary stats: mean/median episodes per tile, detection distribution, valid data pct
Outputs JSON report and prints summary.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
TILL_DATA = ROOT / 'tilling_data'
TILE_DIR = TILL_DATA / 'tiles'
INDEX_PATH = TILL_DATA / 'tile_index.parquet'
EPISODE_INDEX = ROOT / 'embedded_data' / 'episode_index_reclustered.parquet'
FIRE_PATH = ROOT / 'embedded_data' / 'nasa_viirs_with_weather_reclustered.parquet'
REPORT_PATH = TILL_DATA / 'tile_validation_report.json'

if not INDEX_PATH.exists():
    raise SystemExit(f"tile_index.parquet not found at {INDEX_PATH}")

print("Loading tile index ...")
df_tiles = pd.read_parquet(INDEX_PATH)
print(f"Tiles retained: {len(df_tiles)}")

print("Loading episode index ...")
df_epi = pd.read_parquet(EPISODE_INDEX)
valid_episode_ids = set(df_epi['episode_id'].astype(int))

print("Loading fire detections (for recompute) ...")
fire_df = pd.read_parquet(FIRE_PATH)[['episode_id','x','y']]
fire_df['episode_id'] = fire_df['episode_id'].astype(int)

missing_files = []
shape_mismatches = []
na_tiles = []
inf_tiles = []
episode_missing = []
detection_mismatch = []
category_shape_mismatch = []

recomputed_counts = []

# Iterate tiles
for _, tile in tqdm(df_tiles.iterrows(), total=len(df_tiles), desc='Validating tiles'):
    tid = tile['tile_id']
    path = TILE_DIR / f'tile_{tid:04d}.npz'
    if not path.exists():
        missing_files.append(int(tid))
        continue
    data = np.load(path, allow_pickle=True)
    cont = data['continuous_features']  # (C,H,W)
    C, H, W = cont.shape
    # Expect <=256 edges (edges may be truncated)
    if H > 256 or W > 256 or C < 1:
        shape_mismatches.append(int(tid))
    # categorical shapes
    lcm = data.get('lcm_classes')
    fsm = data.get('fsm_classes')
    if lcm is not None and lcm.shape != (H, W):
        category_shape_mismatch.append(int(tid))
    if fsm is not None and fsm.shape != (H, W):
        category_shape_mismatch.append(int(tid))
    # NaN / Inf
    if np.isnan(cont).any():
        na_tiles.append(int(tid))
    if np.isinf(cont).any():
        inf_tiles.append(int(tid))
    # Episode linkage
    for ep_id in tile['episode_ids']:
        if int(ep_id) not in valid_episode_ids:
            episode_missing.append({'tile_id': int(tid), 'episode_id': int(ep_id)})
    # Recompute detection count
    mask = ((fire_df['x'] >= tile['x_start']) & (fire_df['x'] <= tile['x_end']) &
            (fire_df['y'] >= tile['y_start']) & (fire_df['y'] <= tile['y_end']))
    recomputed = int(mask.sum())
    recorded = int(tile['num_detections'])
    if recomputed != recorded:
        detection_mismatch.append({'tile_id': int(tid), 'recorded': recorded, 'recomputed': recomputed})
    recomputed_counts.append(recomputed)

report = {
    'total_tiles': int(len(df_tiles)),
    'missing_files': missing_files,
    'shape_mismatches': shape_mismatches,
    'category_shape_mismatches': category_shape_mismatch,
    'tiles_with_nan': na_tiles,
    'tiles_with_inf': inf_tiles,
    'episode_link_missing': episode_missing,
    'detection_count_mismatches': detection_mismatch,
    'summary': {
        'tiles_ok': int(len(df_tiles) - len(missing_files) - len(shape_mismatches)),
        'mean_recorded_detections': float(df_tiles['num_detections'].mean()),
        'mean_recomputed_detections': float(np.mean(recomputed_counts)) if recomputed_counts else 0.0,
        'mean_episodes_per_tile': float(df_tiles['num_episodes'].mean()),
        'max_episodes_in_tile': int(df_tiles['num_episodes'].max()),
        'mean_valid_data_pct': float(df_tiles['valid_data_pct'].mean()) if 'valid_data_pct' in df_tiles.columns else None
    }
}

with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)

print("\nValidation Summary:")
print(json.dumps(report['summary'], indent=2))
if any([missing_files, shape_mismatches, na_tiles, inf_tiles, episode_missing, detection_mismatch, category_shape_mismatch]):
    print("\nIssues detected:")
    if missing_files: print(f"  Missing files: {len(missing_files)} (first 10: {missing_files[:10]})")
    if shape_mismatches: print(f"  Shape mismatches: {len(shape_mismatches)}")
    if category_shape_mismatch: print(f"  Categorical shape mismatches: {len(category_shape_mismatch)}")
    if na_tiles: print(f"  Tiles with NaN: {len(na_tiles)}")
    if inf_tiles: print(f"  Tiles with Inf: {len(inf_tiles)}")
    if episode_missing: print(f"  Episode IDs missing: {len(episode_missing)}")
    if detection_mismatch: print(f"  Detection count mismatches: {len(detection_mismatch)}")
else:
    print("\nNo issues found. All tiles consistent.")

print(f"\nFull report written to: {REPORT_PATH}")

