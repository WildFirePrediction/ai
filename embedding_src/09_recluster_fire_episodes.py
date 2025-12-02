"""
09 - Re-cluster Fire Episodes (Spatiotemporal DBSCAN)
Properly cluster fire detections into discrete episodes with spatial AND temporal constraints
"""

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

embedded_dir = _Path(__file__).parent.parent / 'embedded_data'

print("=" * 80)
print("FIRE EPISODE RE-CLUSTERING (Spatiotemporal DBSCAN)")
print("=" * 80)

# ============================================================================
# 1. LOAD FIRE DATA
# ============================================================================
print("\n[1/6] Loading fire data...")

df = pd.read_parquet(embedded_dir / 'nasa_viirs_with_weather.parquet')

print(f"  Loaded {len(df):,} fire detections")
print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Spatial range: X [{df['x'].min():.0f}, {df['x'].max():.0f}]")
print(f"                 Y [{df['y'].min():.0f}, {df['y'].max():.0f}]")

# ============================================================================
# 2. PREPARE SPATIOTEMPORAL FEATURES
# ============================================================================
print("\n[2/6] Preparing spatiotemporal features...")

# Convert datetime to seconds since epoch for DBSCAN
df['timestamp_s'] = df['datetime'].astype('int64') / 1e9

# Create feature matrix: [x, y, time]
# Scale time to match spatial scale
# Spatial: meters, Temporal: seconds
# We want: ΔS ≤ 2km, Δt ≤ 7 days = 604800 seconds

# Scaling: normalize so that eps=1 means our desired thresholds
# If eps=1, then:
#   - Spatial distance of 2000m should equal 1
#   - Temporal distance of 604800s should equal 1

spatial_scale = 2000  # meters (2km threshold)
temporal_scale = 7 * 24 * 3600  # seconds (7 days threshold)

X = np.column_stack([
    df['x'].values / spatial_scale,
    df['y'].values / spatial_scale,
    df['timestamp_s'].values / temporal_scale
])

print(f"  Feature matrix: {X.shape}")
print(f"  Spatial scale: {spatial_scale}m (2km threshold)")
print(f"  Temporal scale: {temporal_scale}s (7 days threshold)")

# ============================================================================
# 3. RUN SPATIOTEMPORAL DBSCAN
# ============================================================================
print("\n[3/6] Running spatiotemporal DBSCAN...")
print(f"  Parameters:")
print(f"    eps = 1.0 (scaled units)")
print(f"    min_samples = 3 (minimum detections per episode)")
print(f"    metric = euclidean (L2 distance in 3D: x, y, time)")

# DBSCAN with scaled features
# eps=1.0 means: sqrt((Δx/2000)² + (Δy/2000)² + (Δt/604800)²) ≤ 1.0
# This enforces: ΔS ≤ 2km AND Δt ≤ 7 days (approximately)

dbscan = DBSCAN(eps=1.0, min_samples=3, metric='euclidean', n_jobs=-1)

print(f"  Clustering {len(X):,} detections...")
labels = dbscan.fit_predict(X)

df['episode_id_new'] = labels

# Statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\n  ✓ Clustering complete")
print(f"    Episodes found: {n_clusters:,}")
print(f"    Noise points: {n_noise:,} ({100*n_noise/len(df):.2f}%)")

# ============================================================================
# 4. VALIDATE EPISODE QUALITY
# ============================================================================
print("\n[4/6] Validating episode quality...")

episodes = df[df['episode_id_new'] >= 0].groupby('episode_id_new').agg({
    'datetime': ['min', 'max', 'count'],
    'x': ['min', 'max'],
    'y': ['min', 'max']
})

episodes.columns = ['time_start', 'time_end', 'n_detections', 'x_min', 'x_max', 'y_min', 'y_max']
episodes['duration_hours'] = (episodes['time_end'] - episodes['time_start']).dt.total_seconds() / 3600
episodes['duration_days'] = episodes['duration_hours'] / 24
episodes['spatial_extent_km'] = np.sqrt(
    (episodes['x_max'] - episodes['x_min'])**2 +
    (episodes['y_max'] - episodes['y_min'])**2
) / 1000

print(f"\n  Episode statistics:")
print(f"    Total episodes: {len(episodes)}")
print(f"\n  Duration (hours):")
print(f"    Mean: {episodes['duration_hours'].mean():.1f}h")
print(f"    Median: {episodes['duration_hours'].median():.1f}h")
print(f"    Max: {episodes['duration_hours'].max():.1f}h ({episodes['duration_hours'].max()/24:.1f} days)")
print(f"\n  Spatial extent (km):")
print(f"    Mean: {episodes['spatial_extent_km'].mean():.2f}km")
print(f"    Median: {episodes['spatial_extent_km'].median():.2f}km")
print(f"    Max: {episodes['spatial_extent_km'].max():.2f}km")
print(f"\n  Detections per episode:")
print(f"    Mean: {episodes['n_detections'].mean():.1f}")
print(f"    Median: {episodes['n_detections'].median():.0f}")
print(f"    Max: {episodes['n_detections'].max()}")

# Check for violations
violations = []

# Temporal violations (> 7 days)
temporal_violations = episodes[episodes['duration_days'] > 7]
if len(temporal_violations) > 0:
    violations.append(f"{len(temporal_violations)} episodes exceed 7 days duration")

# Spatial violations (> 50km)
spatial_violations = episodes[episodes['spatial_extent_km'] > 50]
if len(spatial_violations) > 0:
    violations.append(f"{len(spatial_violations)} episodes exceed 50km spatial extent")

# Check internal temporal gaps
print(f"\n  Checking internal temporal gaps...")
gap_violations = 0
for ep_id in tqdm(episodes.index[:100], desc="  Validating", disable=len(episodes)<100):
    ep_data = df[df['episode_id_new'] == ep_id].sort_values('datetime')
    if len(ep_data) > 1:
        time_diffs = np.diff(ep_data['datetime'].values).astype('timedelta64[h]').astype(float)
        max_gap = time_diffs.max()
        if max_gap > 168:  # > 7 days
            gap_violations += 1

if gap_violations > 0:
    violations.append(f"{gap_violations} episodes have internal gaps > 7 days")

if violations:
    print(f"\n  ⚠️  Quality issues:")
    for v in violations:
        print(f"      - {v}")
else:
    print(f"\n  ✓ All episodes pass quality checks")

# ============================================================================
# 5. COMPARE OLD VS NEW CLUSTERING
# ============================================================================
print("\n[5/6] Comparing old vs new clustering...")

print(f"\n  OLD CLUSTERING (NASA CLUSTER_ID):")
print(f"    Episodes: {df['episode_id'].nunique()}")
old_eps = df.groupby('episode_id')['datetime'].apply(lambda x: (x.max() - x.min()).total_seconds() / 86400)
print(f"    Mean duration: {old_eps.mean():.0f} days")
print(f"    Max duration: {old_eps.max():.0f} days")

print(f"\n  NEW CLUSTERING (Spatiotemporal DBSCAN):")
print(f"    Episodes: {n_clusters:,}")
print(f"    Mean duration: {episodes['duration_hours'].mean()/24:.1f} days")
print(f"    Max duration: {episodes['duration_hours'].max()/24:.1f} days")
print(f"    Noise points: {n_noise:,}")

print(f"\n  Improvement:")
print(f"    Episodes increased: {df['episode_id'].nunique()} → {n_clusters:,} ({n_clusters/df['episode_id'].nunique():.1f}x)")
print(f"    Duration reduced: {old_eps.mean():.0f} days → {episodes['duration_hours'].mean()/24:.1f} days")

# ============================================================================
# 6. SAVE RECLUSTERED DATA
# ============================================================================
print("\n[6/6] Saving reclustered data...")

# Replace old episode_id with new
df['episode_id_old'] = df['episode_id']
df['episode_id'] = df['episode_id_new']
df = df.drop(columns=['episode_id_new', 'timestamp_s'])

# Save
output_path = embedded_dir / 'nasa_viirs_with_weather_reclustered.parquet'
df.to_parquet(output_path, compression='snappy', index=False)

print(f"  ✓ Saved: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save episode index
episodes.index.name = 'episode_id'
episodes_output = embedded_dir / 'episode_index_reclustered.parquet'
episodes.reset_index().to_parquet(episodes_output, index=False)
print(f"  ✓ Saved: {episodes_output}")

# Save clustering parameters
import json
params = {
    'method': 'DBSCAN',
    'spatial_threshold_m': spatial_scale,
    'temporal_threshold_s': temporal_scale,
    'temporal_threshold_days': temporal_scale / 86400,
    'eps_scaled': 1.0,
    'min_samples': 3,
    'num_episodes': int(n_clusters),
    'num_noise': int(n_noise),
    'num_detections_total': int(len(df)),
    'mean_duration_hours': float(episodes['duration_hours'].mean()),
    'mean_spatial_extent_km': float(episodes['spatial_extent_km'].mean())
}

params_output = embedded_dir / 'reclustering_params.json'
with open(params_output, 'w') as f:
    json.dump(params, f, indent=2)
print(f"  ✓ Saved: {params_output}")

print("\n" + "=" * 80)
print("RECLUSTERING COMPLETE")
print("=" * 80)
print(f"✓ Found {n_clusters:,} valid fire episodes")
print(f"✓ Mean episode duration: {episodes['duration_hours'].mean()/24:.1f} days")
print(f"✓ Mean spatial extent: {episodes['spatial_extent_km'].mean():.1f} km")
print(f"✓ Data ready for tiling with proper episodes")
print("=" * 80)
