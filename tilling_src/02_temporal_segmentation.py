"""
02 - Temporal Segmentation (Per-episode regions)
For each episode region (extent + padding), create a temporal sequence of fire state and
aggregated weather at the same grid resolution (400m per cell). Each timestep captures
mask/intensity/temp/age and a 5-dim weather vector.
"""

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).parent.parent / 'src').resolve()))

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

script_dir = _Path(__file__).parent
root_dir = script_dir.parent
embedded_dir = root_dir / 'embedded_data'
tilling_dir = root_dir / 'tilling_data'

sequences_dir = tilling_dir / 'sequences'
sequences_dir.mkdir(parents=True, exist_ok=True)

RESOLUTION = 400  # m per cell
PAD = 5           # kept for doc consistency

print("=" * 80)
print("TEMPORAL SEGMENTATION (PER-EPISODE)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/4] Loading regions and fire data...")
regions_df = pd.read_parquet(tilling_dir / 'episode_regions.parquet')
print(f"  Episode regions: {len(regions_df)}")

# Fire detections with weather
fire_df = pd.read_parquet(embedded_dir / 'nasa_viirs_with_weather_reclustered.parquet')
# Episode index
epi_df = pd.read_parquet(embedded_dir / 'episode_index_reclustered.parquet')

# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def create_grids(h, w):
    fire_mask = np.zeros((h, w), dtype=np.uint8)
    fire_int = np.zeros((h, w), dtype=np.float32)
    fire_tmp = np.zeros((h, w), dtype=np.float32)
    fire_age = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)
    return fire_mask, fire_int, fire_tmp, fire_age, counts

# ---------------------------------------------------------------------------
# 3. BUILD SEQUENCES
# ---------------------------------------------------------------------------
print("\n[2/4] Building sequences...")
num_sequences = 0
num_timesteps = 0

for _, reg in tqdm(regions_df.iterrows(), total=len(regions_df), desc='  Episodes'):
    ep_id = int(reg['episode_id'])
    r0, r1, c0, c1 = int(reg['row_start']), int(reg['row_end']), int(reg['col_start']), int(reg['col_end'])
    H, W = r1 - r0, c1 - c0

    # Episode detections within region bounds
    ep_fire = fire_df[(fire_df['episode_id'] == ep_id) &
                      (fire_df['x'] >= (None if np.nan else 0))]  # placeholder to ensure copy
    # Filter by grid coords bounds
    # Convert world to index approximations using episode bbox edges
    # We'll simply filter by world bounds from index convert saved in region NPZ
    reg_npz = np.load(tilling_dir / 'regions' / f'episode_region_{ep_id:05d}.npz', allow_pickle=True)
    wb = reg_npz['world_bounds'].item()
    ep_fire = fire_df[(fire_df['episode_id'] == ep_id) &
                      (fire_df['x'] >= wb['x_min']) & (fire_df['x'] <= wb['x_max']) &
                      (fire_df['y'] >= wb['y_min']) & (fire_df['y'] <= wb['y_max'])].copy()

    if len(ep_fire) == 0:
        continue

    ep_info = epi_df[epi_df['episode_id'] == ep_id].iloc[0]
    t_start, t_end = ep_info['time_start'], ep_info['time_end']

    # 6-hour timesteps across episode
    timesteps = pd.date_range(start=t_start, end=t_end, freq='6h')
    if len(timesteps) < 2:
        continue

    masks = []
    intens = []
    temps = []
    ages = []
    weathers = []
    t_values = []

    # Precompute cell index mapping from world x/y
    # Use saved transform from grid metadata for consistent conversion
    with open(embedded_dir / 'grid_metadata.json', 'r') as f:
        grid_meta = json.load(f)
    a, b, c, d, e, f = grid_meta['transform']

    def to_rc(x, y):
        col = int((x - c) / a)
        row = int((f - y) / abs(e))
        return row, col

    for t in timesteps:
        win_start = t - pd.Timedelta(hours=3)
        win_end = t + pd.Timedelta(hours=3)
        dets = ep_fire[(ep_fire['datetime'] >= win_start) & (ep_fire['datetime'] <= win_end)]

        fm, fi, ft, fa, cnt = create_grids(H, W)

        if len(dets) > 0:
            for _, drow in dets.iterrows():
                rr, cc = to_rc(drow['x'], drow['y'])
                rr -= r0
                cc -= c0
                if 0 <= rr < H and 0 <= cc < W:
                    fm[rr, cc] = 1
                    fi[rr, cc] += float(drow['i'])
                    ft[rr, cc] += float(drow['te'])
                    # age since episode start
                    age_h = (drow['datetime'] - t_start).total_seconds() / 3600.0
                    fa[rr, cc] = max(fa[rr, cc], age_h)
                    cnt[rr, cc] += 1
            nz = cnt > 0
            fi[nz] /= cnt[nz]
            ft[nz] /= cnt[nz]
            weather = np.array([
                dets['w'].mean(), dets['d_x'].mean(), dets['d_y'].mean(), dets['rh'].mean(), dets['r'].mean()
            ], dtype=np.float32)
            weather = np.nan_to_num(weather, nan=0.0)
        else:
            weather = np.zeros(5, dtype=np.float32)

        masks.append(fm)
        intens.append(fi)
        temps.append(ft)
        ages.append(fa)
        weathers.append(weather)
        t_values.append(np.int64(pd.Timestamp(t).value))

    seq = {
        'episode_id': ep_id,
        'grid_coords': {'row_start': r0, 'row_end': r1, 'col_start': c0, 'col_end': c1},
        'timesteps': np.array(t_values, dtype=np.int64),
        'fire_masks': np.stack(masks, axis=0).astype(np.uint8),
        'fire_intensities': np.stack(intens, axis=0).astype(np.float32),
        'fire_temps': np.stack(temps, axis=0).astype(np.float32),
        'fire_ages': np.stack(ages, axis=0).astype(np.float32),
        'weather_states': np.stack(weathers, axis=0).astype(np.float32)
    }

    out = sequences_dir / f'episode_{ep_id:05d}.npz'
    np.savez_compressed(out, **seq)

    num_sequences += 1
    num_timesteps += len(t_values)

print(f"\n  ✓ Sequences: {num_sequences}")
print(f"  Total timesteps: {num_timesteps}")

# ---------------------------------------------------------------------------
# 4. SUMMARY
# ---------------------------------------------------------------------------
summary = {
    'sequences': int(num_sequences),
    'total_timesteps': int(num_timesteps),
    'resolution_m': RESOLUTION,
    'padding_cells': PAD
}
with open(tilling_dir / 'temporal_segments_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved summary: {tilling_dir / 'temporal_segments_summary.json'}")

print("\n" + "=" * 80)
print("TEMPORAL SEGMENTATION COMPLETE (PER-EPISODE)")
print("=" * 80)
