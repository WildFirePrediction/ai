"""
Step 1: Filter VIIRS data for quality long-duration fires.

Criteria:
- At least 5 VIIRS detections
- Spanning at least 3 days (72 hours)
- Proper spatial clustering (within 10km)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json

def load_viirs_data(viirs_path):
    """Load VIIRS fire data."""
    print(f"Loading VIIRS data from {viirs_path}...")
    df = pd.read_csv(viirs_path)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
                                     format='%Y-%m-%d %H%M')

    # Sort by time
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"Loaded {len(df):,} VIIRS detections")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def cluster_fires_spatiotemporal(df, spatial_eps_km=10, temporal_eps_hours=48):
    """
    Cluster fires using spatial-temporal proximity.

    Args:
        spatial_eps_km: Max distance between points in same fire (km)
        temporal_eps_hours: Max time gap between points in same fire (hours)
    """
    print(f"\nClustering fires (spatial={spatial_eps_km}km, temporal={temporal_eps_hours}h)...")

    # Convert to numpy arrays
    # Handle both uppercase and lowercase column names
    lat_col = 'LATITUDE' if 'LATITUDE' in df.columns else 'latitude'
    lon_col = 'LONGITUDE' if 'LONGITUDE' in df.columns else 'longitude'
    coords = df[[lat_col, lon_col]].values
    times = df['datetime'].values

    # Normalize spatial coordinates (rough: 1 degree ≈ 111km at equator)
    coords_km = coords * 111.0

    # Normalize temporal (convert to hours since start)
    time_start = times.min()
    times_hours = (times - time_start) / np.timedelta64(1, 'h')

    # Create feature matrix: [lat_km, lon_km, time_hours]
    # Scale time to match spatial scale
    time_scale = spatial_eps_km / temporal_eps_hours
    features = np.column_stack([coords_km, times_hours * time_scale])

    # DBSCAN clustering
    eps = spatial_eps_km
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(features)

    print(f"Found {labels.max() + 1} fire clusters (excluding noise)")
    print(f"Noise points: {(labels == -1).sum():,} ({100*(labels == -1).sum()/len(labels):.1f}%)")

    return labels


def filter_quality_fires(df, labels, min_detections=5, min_duration_days=3):
    """
    Filter for quality fires based on criteria.

    Args:
        min_detections: Minimum number of detections
        min_duration_days: Minimum duration in days
    """
    print(f"\nFiltering for quality fires (min {min_detections} detections, {min_duration_days} days)...")

    df = df.copy()
    df['fire_cluster'] = labels

    # Handle both uppercase and lowercase column names
    lat_col = 'LATITUDE' if 'LATITUDE' in df.columns else 'latitude'
    lon_col = 'LONGITUDE' if 'LONGITUDE' in df.columns else 'longitude'

    # Get cluster statistics
    cluster_stats = []
    for cluster_id in range(labels.max() + 1):
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]

        time_min = cluster_df['datetime'].min()
        time_max = cluster_df['datetime'].max()
        duration_days = (time_max - time_min).total_seconds() / 86400

        stats = {
            'cluster_id': cluster_id,
            'num_detections': cluster_mask.sum(),
            'duration_days': duration_days,
            'start_time': time_min,
            'end_time': time_max,
            'center_lat': cluster_df[lat_col].mean(),
            'center_lon': cluster_df[lon_col].mean()
        }
        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)

    # Apply filters
    valid_clusters = stats_df[
        (stats_df['num_detections'] >= min_detections) &
        (stats_df['duration_days'] >= min_duration_days)
    ]['cluster_id'].values

    print(f"\nCluster statistics:")
    print(f"  Total clusters: {len(stats_df)}")
    print(f"  Valid clusters: {len(valid_clusters)} ({100*len(valid_clusters)/len(stats_df):.1f}%)")

    # Filter dataframe
    valid_mask = df['fire_cluster'].isin(valid_clusters)
    filtered_df = df[valid_mask].copy()

    # Remap cluster IDs to be contiguous
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_clusters)}
    filtered_df['fire_cluster'] = filtered_df['fire_cluster'].map(cluster_mapping)

    print(f"\nFiltered data:")
    print(f"  Detections: {len(df):,} → {len(filtered_df):,} ({100*len(filtered_df)/len(df):.1f}%)")
    print(f"  Fire clusters: {len(stats_df)} → {len(valid_clusters)}")

    # Print statistics of filtered fires
    filtered_stats = stats_df[stats_df['cluster_id'].isin(valid_clusters)]
    print(f"\nFiltered fire statistics:")
    print(f"  Detections per fire - Mean: {filtered_stats['num_detections'].mean():.1f}, "
          f"Median: {filtered_stats['num_detections'].median():.0f}, "
          f"Max: {filtered_stats['num_detections'].max():.0f}")
    print(f"  Duration (days) - Mean: {filtered_stats['duration_days'].mean():.1f}, "
          f"Median: {filtered_stats['duration_days'].median():.1f}, "
          f"Max: {filtered_stats['duration_days'].max():.1f}")

    return filtered_df, filtered_stats


def main():
    # Paths
    repo_root = Path('/home/chaseungjoon/code/WildfirePrediction')
    viirs_path = repo_root / 'embedded_data' / 'nasa_viirs_embedded.parquet'
    output_dir = repo_root / 'data' / 'filtered_fires'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"Loading VIIRS data from {viirs_path}...")
    df = pd.read_parquet(viirs_path)

    # Parse datetime if needed
    if 'datetime' not in df.columns:
        if 'acq_date' in df.columns and 'acq_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
                                             format='%Y-%m-%d %H%M')
        else:
            raise ValueError("Cannot find datetime or acq_date/acq_time columns")

    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"Loaded {len(df):,} VIIRS detections")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Continue with original logic
    # df = load_viirs_data(viirs_path)  # Commented out, using parquet instead

    # Cluster fires
    labels = cluster_fires_spatiotemporal(df, spatial_eps_km=10, temporal_eps_hours=48)

    # Filter for quality
    filtered_df, stats_df = filter_quality_fires(
        df, labels,
        min_detections=5,
        min_duration_days=3
    )

    # Save filtered data
    output_viirs = output_dir / 'filtered_viirs.parquet'
    output_stats = output_dir / 'fire_cluster_stats.csv'

    # Drop the fire_cluster column before saving VIIRS data
    filtered_df_save = filtered_df.drop(columns=['fire_cluster'])
    filtered_df_save.to_parquet(output_viirs, index=False)
    stats_df.to_csv(output_stats, index=False)

    print(f"\n✓ Saved filtered VIIRS data: {output_viirs}")
    print(f"✓ Saved cluster statistics: {output_stats}")

    # Save cluster mapping for reference
    cluster_map_path = output_dir / 'cluster_mapping.json'
    cluster_mapping = {}
    for _, row in filtered_df.iterrows():
        idx = int(row.name)
        cluster_id = int(row['fire_cluster'])
        cluster_mapping[idx] = cluster_id

    with open(cluster_map_path, 'w') as f:
        json.dump({
            'num_clusters': int(filtered_df['fire_cluster'].max() + 1),
            'mapping': cluster_mapping
        }, f, indent=2)

    print(f"✓ Saved cluster mapping: {cluster_map_path}")


if __name__ == '__main__':
    main()
