"""
03 - Environment Assembly (Per-window agent)
Build RL-ready environments where:
- One agent per sliding window observes fire progression
- Observation per timestep: [static (3+2), fire (4), weather (5)]
- Action space: 9 directions (N, NE, E, SE, S, SW, W, NW, NONE)

Saves pickled dict per environment and a manifest parquet.
"""

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).parent.parent / 'src').resolve()))

import numpy as np
import pandas as pd
import json
import pickle
from tqdm import tqdm

script_dir = _Path(__file__).parent
root_dir = script_dir.parent
embedded_dir = root_dir / 'embedded_data'
tilling_dir = root_dir / 'tilling_data'

env_dir = tilling_dir / 'environments'
env_dir.mkdir(parents=True, exist_ok=True)

ACTIONS_9 = ['N','NE','E','SE','S','SW','W','NW','NONE']
ACTION_TO_DELTA = {
    'N': (-1, 0), 'NE': (-1, 1), 'E': (0, 1), 'SE': (1, 1),
    'S': (1, 0), 'SW': (1, -1), 'W': (0, -1), 'NW': (-1, -1), 'NONE': (0, 0)
}

print("=" * 80)
print("ENVIRONMENT ASSEMBLY (PER-WINDOW)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/4] Loading static crops and sequences...")
regions_df = pd.read_parquet(tilling_dir / 'window_regions.parquet')
seq_files = sorted((tilling_dir / 'sequences').glob('window_*.npz'))
print(f"  Regions: {len(regions_df)} | Sequences: {len(seq_files)}")

with open(embedded_dir / 'grid_metadata.json', 'r') as f:
    grid_meta = json.load(f)

# ---------------------------------------------------------------------------
# 2. BUILD ENVIRONMENTS
# ---------------------------------------------------------------------------
print("\n[2/4] Building environments...")
manifest = []
count = 0

for seq_path in tqdm(seq_files, desc='  Windows'):
    win_id = int(seq_path.stem.split('_')[1])

    reg_path = tilling_dir / 'regions' / f'window_region_{win_id:05d}.npz'
    if not reg_path.exists():
        continue

    reg = np.load(reg_path, allow_pickle=True)
    seq = np.load(seq_path, allow_pickle=True)

    static_cont = reg['continuous_features']  # (C, H, W) C ~ [dem, rsp, ndvi]
    H, W = static_cont.shape[1], static_cont.shape[2]

    # Build an invalid mask for static continuous (NaNs) and fill with 0
    invalid_mask = np.any(np.isnan(static_cont), axis=0).astype(np.uint8)  # (H,W)
    static_cont = np.nan_to_num(static_cont, nan=0.0)

    # Categorical maps
    lcm = reg.get('lcm_classes')
    fsm = reg.get('fsm_classes')
    if lcm is None:
        lcm = np.zeros((H, W), dtype=np.uint16)
    if fsm is None:
        fsm = np.zeros((H, W), dtype=np.uint16)

    # Temporal
    fire_masks = seq['fire_masks']       # (T, H, W)
    fire_intensities = seq['fire_intensities']
    fire_temps = seq['fire_temps']
    fire_ages = seq['fire_ages']
    weather_states = seq['weather_states']  # (T, 5)
    timesteps = seq['timesteps']
    T = fire_masks.shape[0]

    # CRITICAL: Validate temporal and static dimensions match!
    if fire_masks.shape[1:] != (H, W):
        print(f"  WARNING: Skipping env {win_id} - dimension mismatch!")
        print(f"    Static: ({H}, {W}) | Temporal: {fire_masks.shape[1:]}")
        continue

    # Build observation tensor lazily in training; here store components
    env = {
        'window_id': win_id,
        'grid_coords': reg['grid_coords'].item(),
        'world_bounds': reg['world_bounds'].item(),
        'static': {
            'continuous': static_cont.astype(np.float32),
            'lcm': lcm.astype(np.uint16),
            'fsm': fsm.astype(np.uint16),
            'feature_names': reg['feature_names'],
            'invalid_mask': invalid_mask  # 1 where any static channel was NaN
        },
        'temporal': {
            'timesteps': timesteps,
            'fire_masks': fire_masks.astype(np.uint8),
            'fire_intensities': fire_intensities.astype(np.float32),
            'fire_temps': fire_temps.astype(np.float32),
            'fire_ages': fire_ages.astype(np.float32),
            'weather_states': weather_states.astype(np.float32)
        },
        'agent': {
            'action_space': ACTIONS_9,
            'action_to_delta': ACTION_TO_DELTA,
            'initial_position_rule': 'first_detection_pixel'
        },
        'metadata': {
            'resolution_m': 400,
            'num_timesteps': int(T),
            'height': int(H),
            'width': int(W),
            'crs': grid_meta['crs']
        }
    }

    out_file = env_dir / f'env_{win_id:05d}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest.append({
        'env_id': f'env_{win_id:05d}',
        'window_id': win_id,
        'num_timesteps': int(T),
        'height': int(H),
        'width': int(W),
        'file_path': str(out_file.relative_to(tilling_dir))
    })
    count += 1

print(f"  ✓ Environments built: {count}")

# ---------------------------------------------------------------------------
# 3. SAVE MANIFEST + SUMMARY
# ---------------------------------------------------------------------------
print("\n[3/4] Saving manifest and summary...")

df_manifest = pd.DataFrame(manifest)
manifest_path = env_dir / 'environment_manifest.parquet'
df_manifest.to_parquet(manifest_path, index=False)
print(f"  Saved manifest: {manifest_path}")

# Summary
summary = {
    'environments': int(len(df_manifest)),
    'mean_timesteps': float(df_manifest['num_timesteps'].mean()) if len(df_manifest) else 0,
    'mean_size_cells': {
        'height': float(df_manifest['height'].mean()) if len(df_manifest) else 0,
        'width': float(df_manifest['width'].mean()) if len(df_manifest) else 0
    },
    'action_space': ACTIONS_9,
}
with open(tilling_dir / 'environment_assembly_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved summary: {tilling_dir / 'environment_assembly_summary.json'}")

print("\n" + "=" * 80)
print("ENVIRONMENT ASSEMBLY COMPLETE (PER-EPISODE)")
print("=" * 80)
