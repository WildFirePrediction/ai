"""
01 - NASA VIIRS Fire Data Embedding
Processes NASA VIIRS wildfire hotspot data using pre-clustered fire archives
"""

import os
import numpy as np
import pandas as pd
import pyproj
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Create output directory
output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("NASA VIIRS FIRE DATA EMBEDDING")
print("=" * 80)

# ============================================================================
# 1. LOAD PRE-CLUSTERED FIRE DATA
# ============================================================================
print("\n[1/8] Loading pre-clustered fire data...")

nasa_dir = Path('../data/NASA/VIIRS')
csv_files = sorted(nasa_dir.glob('*/clustered_fire_archive_*.csv'))

print(f"Found {len(csv_files)} clustered fire archive files")
for f in csv_files[:3]:
    print(f"  - {f.parent.name}/{f.name}")

# Load all clustered fire data
dfs = []
for csv_file in tqdm(csv_files, desc="Loading CSV files"):
    try:
        # Clustered files may have header row, check first line
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()

        # If first line contains column names, skip it
        if 'LATITUDE' in first_line.upper() or 'latitude' in first_line.lower():
            df = pd.read_csv(csv_file, skiprows=1, header=None, names=[
                'LATITUDE', 'LONGITUDE', 'BRIGHTNESS', 'SCAN', 'TRACK',
                'ACQ_DATE', 'ACQ_TIME', 'SATELLITE', 'INSTRUMENT', 'CONFIDENCE',
                'VERSION', 'BRIGHT_T31', 'FRP', 'DAYNIGHT', 'TYPE', 'geometry', 'CLUSTER_ID'
            ])
        else:
            df = pd.read_csv(csv_file, header=None, names=[
                'LATITUDE', 'LONGITUDE', 'BRIGHTNESS', 'SCAN', 'TRACK',
                'ACQ_DATE', 'ACQ_TIME', 'SATELLITE', 'INSTRUMENT', 'CONFIDENCE',
                'VERSION', 'BRIGHT_T31', 'FRP', 'DAYNIGHT', 'TYPE', 'geometry', 'CLUSTER_ID'
            ])
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {csv_file.name}: {e}")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\nTotal records: {len(df_raw):,}")
print(f"Pre-defined clusters: {df_raw['CLUSTER_ID'].nunique():,}")
print(f"\nColumns: {df_raw.columns.tolist()}")

# ============================================================================
# 2. PRE-CONFIG: FILTER LOW CONFIDENCE DATA
# ============================================================================
print("\n[2/8] Filtering low confidence observations...")

print(f"Before filtering: {len(df_raw):,} records")
print(f"Confidence distribution:")
print(df_raw['CONFIDENCE'].value_counts())

# Remove low confidence (l = low)
df_filtered = df_raw[df_raw['CONFIDENCE'] != 'l'].copy()

print(f"\nAfter filtering: {len(df_filtered):,} records")
print(f"Removed {len(df_raw) - len(df_filtered):,} low confidence records")

# ============================================================================
# 3. COORDINATE TRANSFORMATION: WGS84 → EPSG:5179
# ============================================================================
print("\n[3/8] Transforming coordinates to EPSG:5179...")

# Use modern Transformer API (not deprecated transform)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

# Transform lon, lat to x, y
x, y = transformer.transform(
    df_filtered['LONGITUDE'].values,
    df_filtered['LATITUDE'].values
)

df_filtered['x'] = x
df_filtered['y'] = y

print(f"Coordinate transformation completed")
print(f"X range: [{df_filtered['x'].min():.2f}, {df_filtered['x'].max():.2f}] meters")
print(f"Y range: [{df_filtered['y'].min():.2f}, {df_filtered['y'].max():.2f}] meters")

# ============================================================================
# 4. TEMPERATURE FEATURE (te)
# ============================================================================
print("\n[4/8] Processing temperature feature...")

# Convert BRIGHT_T31 from Kelvin to Celsius
df_filtered['te'] = df_filtered['BRIGHT_T31'] - 273.15

print(f"Temperature statistics:")
print(df_filtered['te'].describe())

# ============================================================================
# 5. FIRE INTENSITY FEATURE (i)
# ============================================================================
print("\n[5/8] Processing fire intensity feature...")

# Combine FRP and BRIGHTNESS
# i = α * FRP + β * (BRIGHTNESS - 273.15)
alpha = 1.0
beta = 0.1

df_filtered['i_raw'] = (
    alpha * df_filtered['FRP'] +
    beta * (df_filtered['BRIGHTNESS'] - 273.15)
)

# Apply log scaling: i' = log(1 + i)
df_filtered['i'] = np.log1p(df_filtered['i_raw'])

print(f"Fire Intensity statistics:")
print(df_filtered[['FRP', 'i_raw', 'i']].describe())

# ============================================================================
# 6. TEMPORAL FEATURE (tm) - USING PRE-CLUSTERED EPISODES
# ============================================================================
print("\n[6/8] Processing temporal features using pre-clustered episodes...")

# Parse datetime
df_filtered['datetime'] = pd.to_datetime(
    df_filtered['ACQ_DATE'] + ' ' + df_filtered['ACQ_TIME'].astype(str).str.zfill(4),
    format='%Y-%m-%d %H%M'
)

# Sort by datetime
df_filtered = df_filtered.sort_values('datetime').reset_index(drop=True)

# Use the pre-existing CLUSTER_ID as episode_id
df_filtered['episode_id'] = df_filtered['CLUSTER_ID']

n_episodes = df_filtered['episode_id'].nunique()
print(f"Total fire episodes (from clustering): {n_episodes:,}")

# Calculate elapsed time for each fire point within its episode
def calculate_elapsed_time(group):
    if len(group) == 0:
        return group
    t0 = group['datetime'].min()
    group['tm'] = (group['datetime'] - t0).dt.total_seconds() / 3600.0  # hours
    return group

df_filtered = df_filtered.groupby('episode_id', group_keys=False).apply(
    calculate_elapsed_time
)

print(f"Time elapsed (tm) statistics:")
print(df_filtered['tm'].describe())

# Episode statistics
episode_stats = df_filtered.groupby('episode_id').agg({
    'datetime': ['min', 'max', 'count'],
    'i': 'mean',
    'FRP': 'max'
}).reset_index()
episode_stats.columns = ['episode_id', 'start', 'end', 'detections', 'mean_intensity', 'max_frp']
episode_stats['duration_hours'] = (episode_stats['end'] - episode_stats['start']).dt.total_seconds() / 3600

print(f"\nEpisode statistics:")
print(f"  Mean duration: {episode_stats['duration_hours'].mean():.2f} hours")
print(f"  Mean detections per episode: {episode_stats['detections'].mean():.1f}")
print(f"  Episodes with >10 detections: {(episode_stats['detections'] > 10).sum():,}")

# ============================================================================
# 7. NORMALIZATION
# ============================================================================
print("\n[7/8] Normalizing features...")

# Normalize x, y using z-score
df_filtered['x_norm'] = (df_filtered['x'] - df_filtered['x'].mean()) / df_filtered['x'].std()
df_filtered['y_norm'] = (df_filtered['y'] - df_filtered['y'].mean()) / df_filtered['y'].std()

# Normalize temperature using z-score
df_filtered['te_norm'] = (df_filtered['te'] - df_filtered['te'].mean()) / df_filtered['te'].std()

# Normalize intensity using z-score
df_filtered['i_norm'] = (df_filtered['i'] - df_filtered['i'].mean()) / df_filtered['i'].std()

# Normalize time elapsed using log and z-score
df_filtered['tm_log'] = np.log1p(df_filtered['tm'])
df_filtered['tm_norm'] = (df_filtered['tm_log'] - df_filtered['tm_log'].mean()) / df_filtered['tm_log'].std()

print("Normalization statistics:")
for col in ['x_norm', 'y_norm', 'te_norm', 'i_norm', 'tm_norm']:
    print(f"  {col}: mean={df_filtered[col].mean():.6f}, std={df_filtered[col].std():.6f}")

# ============================================================================
# 8. SAVE EMBEDDED DATA
# ============================================================================
print("\n[8/8] Saving embedded data...")

# Select columns for output
output_cols = [
    'episode_id', 'datetime',
    'LONGITUDE', 'LATITUDE',
    'x', 'y', 'x_norm', 'y_norm',
    'te', 'te_norm',
    'i', 'i_norm',
    'tm', 'tm_norm',
    'FRP', 'BRIGHTNESS', 'BRIGHT_T31', 'CONFIDENCE'
]

df_output = df_filtered[output_cols].copy()

# Save to parquet
output_path = output_dir / 'nasa_viirs_embedded.parquet'
df_output.to_parquet(output_path, index=False)

print(f"Saved embedded data to: {output_path}")
print(f"Shape: {df_output.shape}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save normalization statistics
norm_stats = {
    'x_mean': float(df_filtered['x'].mean()),
    'x_std': float(df_filtered['x'].std()),
    'y_mean': float(df_filtered['y'].mean()),
    'y_std': float(df_filtered['y'].std()),
    'te_mean': float(df_filtered['te'].mean()),
    'te_std': float(df_filtered['te'].std()),
    'i_mean': float(df_filtered['i'].mean()),
    'i_std': float(df_filtered['i'].std()),
    'tm_log_mean': float(df_filtered['tm_log'].mean()),
    'tm_log_std': float(df_filtered['tm_log'].std()),
    'n_episodes': int(n_episodes),
    'n_records': int(len(df_output))
}

stats_path = output_dir / 'nasa_viirs_norm_stats.json'
with open(stats_path, 'w') as f:
    json.dump(norm_stats, f, indent=2)

print(f"Saved normalization stats to: {stats_path}")

# Save episode index
episode_path = output_dir / 'nasa_viirs_episode_index.parquet'
episode_stats.to_parquet(episode_path, index=False)
print(f"Saved episode index to: {episode_path}")

print("\n" + "=" * 80)
print("NASA VIIRS EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Processed {len(df_output):,} fire detections")
print(f"✓ Identified {n_episodes:,} fire episodes")
print(f"✓ Ready for RL training")

