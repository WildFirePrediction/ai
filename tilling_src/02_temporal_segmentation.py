"""
02 - Temporal Segmentation (Per-window regions)
For each window region (extent + padding), create a temporal sequence of fire state and
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
print("TEMPORAL SEGMENTATION (PER-WINDOW)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/4] Loading regions and fire data...")
regions_df = pd.read_parquet(tilling_dir / 'window_regions.parquet')
print(f"  Window regions: {len(regions_df)}")

# Fire detections with weather
fire_df = pd.read_parquet(embedded_dir / 'nasa_viirs_with_weather.parquet')
# Window index
windows_df = pd.read_parquet(embedded_dir / 'sliding_windows_index.parquet')

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

for _, reg in tqdm(regions_df.iterrows(), total=len(regions_df), desc='  Windows'):
    win_id = int(reg['window_id'])
    r0, r1, c0, c1 = int(reg['row_start']), int(reg['row_end']), int(reg['col_start']), int(reg['col_end'])
    H, W = r1 - r0, c1 - c0

    # Window detections within region bounds
    # Load region bounds from saved NPZ
    reg_npz = np.load(tilling_dir / 'regions' / f'window_region_{win_id:05d}.npz', allow_pickle=True)
    wb = reg_npz['world_bounds'].item()

    # Get window info from index
    win_info = windows_df[windows_df['window_id'] == win_id].iloc[0]
    t_start, t_end = win_info['time_start'], win_info['time_end']

    # Filter fire detections by spatial and temporal bounds
    win_fire = fire_df[
        (fire_df['datetime'] >= t_start) &
        (fire_df['datetime'] <= t_end) &
        (fire_df['x'] >= wb['x_min']) & (fire_df['x'] <= wb['x_max']) &
        (fire_df['y'] >= wb['y_min']) & (fire_df['y'] <= wb['y_max'])
    ].copy()

    if len(win_fire) == 0:
        continue

    # 2-hour timesteps across episode (CHANGED from 6h for better density)
    timesteps = pd.date_range(start=t_start, end=t_end, freq='2h')

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
        win_start = t - pd.Timedelta(hours=3)  # ±3h window for better fire continuity
        win_end = t + pd.Timedelta(hours=3)
        dets = win_fire[(win_fire['datetime'] >= win_start) & (win_fire['datetime'] <= win_end)]

        # SKIP TIMESTEPS WITH NO FIRE (prevents sparse/discontinuous sequences)
        if len(dets) == 0:
            continue

        fm, fi, ft, fa, cnt = create_grids(H, W)

        for _, drow in dets.iterrows():
            rr, cc = to_rc(drow['x'], drow['y'])
            rr -= r0
            cc -= c0
            if 0 <= rr < H and 0 <= cc < W:
                fm[rr, cc] = 1
                fi[rr, cc] += float(drow['i'])
                ft[rr, cc] += float(drow['BRIGHTNESS'])  # Use fire brightness temperature
                # age since episode start
                age_h = (drow['datetime'] - t_start).total_seconds() / 3600.0
                fa[rr, cc] = max(fa[rr, cc], age_h)
                cnt[rr, cc] += 1
        nz = cnt > 0
        fi[nz] /= cnt[nz]
        ft[nz] /= cnt[nz]
        # Weather: [temp, humidity, wind_speed, wind_x, wind_y, rainfall]
        # Use available columns or defaults
        weather = np.array([
            dets['te'].mean() if 'te' in dets.columns else 0.0,
            dets['rh'].mean() if 'rh' in dets.columns else 0.0,
            dets['w'].mean() if 'w' in dets.columns else 0.0,
            dets['d_x'].mean() if 'd_x' in dets.columns else 0.0,
            dets['d_y'].mean() if 'd_y' in dets.columns else 0.0,
            dets['r'].mean() if 'r' in dets.columns else 0.0
        ], dtype=np.float32)
        weather = np.nan_to_num(weather, nan=0.0)

        masks.append(fm)
        intens.append(fi)
        temps.append(ft)
        ages.append(fa)
        weathers.append(weather)
        t_values.append(np.int64(pd.Timestamp(t).value))

    # Skip windows with too few fire timesteps (need at least 3 for learning)
    if len(masks) < 3:
        continue

    seq = {
        'window_id': win_id,
        'grid_coords': {'row_start': r0, 'row_end': r1, 'col_start': c0, 'col_end': c1},
        'timesteps': np.array(t_values, dtype=np.int64),
        'fire_masks': np.stack(masks, axis=0).astype(np.uint8),
        'fire_intensities': np.stack(intens, axis=0).astype(np.float32),
        'fire_temps': np.stack(temps, axis=0).astype(np.float32),
        'fire_ages': np.stack(ages, axis=0).astype(np.float32),
        'weather_states': np.stack(weathers, axis=0).astype(np.float32)
    }

    out = sequences_dir / f'window_{win_id:05d}.npz'
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
print("TEMPORAL SEGMENTATION COMPLETE (PER-WINDOW)")
print("=" * 80)
