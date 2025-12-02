"""
Re-cluster fires into REAL episodes using spatio-temporal DBSCAN
Max temporal gap: 48 hours, Max spatial distance: 10km
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path
import sys

def spatiotemporal_clustering(df, eps_spatial_km=10, eps_temporal_hours=48, min_samples=4):
    """
    Cluster fires using spatio-temporal distance
    
    Args:
        df: DataFrame with columns ['datetime', 'x', 'y', 'LATITUDE', 'LONGITUDE']
        eps_spatial_km: Max spatial distance in km
        eps_temporal_hours: Max temporal gap in hours
        min_samples: Min cluster size
    """
    # Convert datetime to hours since first detection
    df = df.copy()
    df['time_hours'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds() / 3600
    
    # Normalize coordinates and time for DBSCAN
    # Scale: 1 hour = 1 unit, 1 km = 1 unit
    X = np.column_stack([
        df['x'].values / 1000,  # Convert m to km
        df['y'].values / 1000,
        df['time_hours'].values
    ])
    
    # DBSCAN with weighted distance
    # eps controls the max distance in the normalized space
    eps = np.sqrt(eps_spatial_km**2 + eps_temporal_hours**2) / np.sqrt(3)
    
    print(f"Running DBSCAN with eps={eps:.2f}, min_samples={min_samples}")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(X)
    
    return labels

def main():
    print("="*80)
    print("RE-CLUSTERING FIRES INTO REAL EPISODES")
    print("="*80)
    
    # Load NASA VIIRS data
    nasa_file = Path('embedded_data/nasa_viirs_embedded.parquet')
    print(f"\nLoading {nasa_file}...")
    df = pd.read_parquet(nasa_file)
    
    print(f"Total detections: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Remove old episode_id
    if 'episode_id' in df.columns:
        df = df.drop(columns=['episode_id'])
    
    # Cluster with spatio-temporal constraints
    print("\nClustering with spatio-temporal constraints:")
    print("  Max spatial distance: 10 km")
    print("  Max temporal gap: 48 hours")
    print("  Min cluster size: 4 detections")
    
    labels = spatiotemporal_clustering(df, eps_spatial_km=10, eps_temporal_hours=48, min_samples=4)
    
    # Add new episode_id
    df['episode_id'] = labels
    
    # Filter out noise (label = -1)
    valid_episodes = df[df['episode_id'] >= 0].copy()
    
    print(f"\nClustering results:")
    print(f"  Total clusters found: {labels.max() + 1}")
    print(f"  Noise points (filtered): {(labels == -1).sum():,}")
    print(f"  Valid detections: {len(valid_episodes):,}")
    
    # Check episode statistics
    episode_stats = valid_episodes.groupby('episode_id').agg({
        'datetime': ['count', 'min', 'max'],
        'LATITUDE': 'mean',
        'LONGITUDE': 'mean'
    })
    
    episode_stats['duration_hours'] = (
        (episode_stats[('datetime', 'max')] - episode_stats[('datetime', 'min')]).dt.total_seconds() / 3600
    )
    episode_stats['num_timesteps'] = valid_episodes.groupby('episode_id')['datetime'].nunique()
    
    print(f"\nEpisode statistics:")
    print(f"  Total episodes: {len(episode_stats)}")
    print(f"  Detections per episode: {episode_stats[('datetime', 'count')].mean():.1f} (mean)")
    print(f"  Timesteps per episode: {episode_stats['num_timesteps'].mean():.1f} (mean)")
    print(f"  Duration: {episode_stats['duration_hours'].mean():.1f} hours (mean)")
    
    # Filter episodes with >= 4 timesteps
    valid_episode_ids = episode_stats[episode_stats['num_timesteps'] >= 4].index
    final_df = valid_episodes[valid_episodes['episode_id'].isin(valid_episode_ids)].copy()
    
    # Renumber episodes sequentially
    episode_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(valid_episode_ids))}
    final_df['episode_id'] = final_df['episode_id'].map(episode_mapping)
    
    print(f"\nAfter filtering (>= 4 timesteps):")
    print(f"  Episodes: {len(valid_episode_ids)}")
    print(f"  Detections: {len(final_df):,}")
    
    # Show sample episodes
    print(f"\nSample episodes (first 5):")
    for ep_id in range(min(5, len(valid_episode_ids))):
        ep = final_df[final_df['episode_id'] == ep_id]
        ts_count = ep['datetime'].nunique()
        duration = (ep['datetime'].max() - ep['datetime'].min()).total_seconds() / 3600
        print(f"  Ep {ep_id}: {len(ep):4d} detections, {ts_count:2d} timesteps, "
              f"{duration:6.1f} hours ({ep['datetime'].min()} to {ep['datetime'].max()})")
    
    # Save updated data
    output_file = Path('embedded_data/nasa_viirs_reclustered.parquet')
    final_df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Save episode index
    episode_index = final_df.groupby('episode_id').agg({
        'datetime': ['min', 'max', 'count'],
        'LATITUDE': 'mean',
        'LONGITUDE': 'mean',
        'x': 'mean',
        'y': 'mean'
    }).reset_index()
    
    episode_index.columns = ['episode_id', 'start', 'end', 'detections', 'lat', 'lon', 'x', 'y']
    episode_index['duration_hours'] = (episode_index['end'] - episode_index['start']).dt.total_seconds() / 3600
    episode_index['num_timesteps'] = final_df.groupby('episode_id')['datetime'].nunique().values
    
    index_file = Path('embedded_data/nasa_viirs_episode_index_reclustered.parquet')
    episode_index.to_parquet(index_file, index=False)
    print(f"Saved episode index to {index_file}")
    
    print("\n" + "="*80)
    print("RE-CLUSTERING COMPLETE")
    print("="*80)
    
    return final_df, episode_index

if __name__ == '__main__':
    df, index = main()
