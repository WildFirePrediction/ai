"""
Step 2: Create sliding windows from filtered VIIRS data.

Takes filtered fire detections and creates 48-hour sliding windows.
This replaces the embedded_data/sliding_windows_index.parquet file.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import json
from tqdm import tqdm

def create_sliding_windows(df_fire, window_hours=48, stride_hours=24):
    """
    Create sliding windows from fire detections.

    Args:
        df_fire: DataFrame with fire detections
        window_hours: Window duration in hours
        stride_hours: Stride between windows

    Returns:
        DataFrame with window definitions
    """
    print(f"\nCreating sliding windows ({window_hours}h window, {stride_hours}h stride)...")

    # Sort by time
    df_fire = df_fire.sort_values('datetime').reset_index(drop=True)

    time_min = df_fire['datetime'].min()
    time_max = df_fire['datetime'].max()

    print(f"  Fire data timespan: {time_min} to {time_max}")
    print(f"  Duration: {(time_max - time_min).total_seconds() / 86400:.1f} days")

    # Generate window start times
    window_duration = timedelta(hours=window_hours)
    stride_duration = timedelta(hours=stride_hours)

    windows = []
    window_id = 0
    current_time = time_min

    while current_time + window_duration <= time_max:
        window_end = current_time + window_duration

        # Get detections in this window
        mask = (df_fire['datetime'] >= current_time) & (df_fire['datetime'] < window_end)
        window_fires = df_fire[mask]

        if len(window_fires) > 0:
            # Compute spatial extent (in projected coords)
            x_min = window_fires['x'].min()
            x_max = window_fires['x'].max()
            y_min = window_fires['y'].min()
            y_max = window_fires['y'].max()

            windows.append({
                'window_id': window_id,
                'time_start': current_time,
                'time_end': window_end,
                'num_detections': len(window_fires),
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            })
            window_id += 1

        current_time += stride_duration

    df_windows = pd.DataFrame(windows)

    # Add duration_hours column (needed by tiling script)
    df_windows['duration_hours'] = (df_windows['time_end'] - df_windows['time_start']).dt.total_seconds() / 3600

    # Add spatial_extent_km (needed by tiling script)
    # Compute diagonal extent from bounding box (x,y are in meters)
    df_windows['spatial_extent_km'] = np.sqrt(
        (df_windows['x_max'] - df_windows['x_min'])**2 +
        (df_windows['y_max'] - df_windows['y_min'])**2
    ) / 1000.0

    print(f"\n  Created {len(df_windows)} windows")
    print(f"  Detections per window - Mean: {df_windows['num_detections'].mean():.1f}, "
          f"Median: {df_windows['num_detections'].median():.0f}, "
          f"Max: {df_windows['num_detections'].max()}")

    return df_windows


def main():
    repo_root = Path('/home/chaseungjoon/code/WildfirePrediction')

    # Load filtered VIIRS data
    filtered_dir = repo_root / 'data' / 'filtered_fires'
    viirs_path = filtered_dir / 'filtered_viirs.parquet'

    print("Loading filtered VIIRS data...")
    df_fire = pd.read_parquet(viirs_path)

    # Parse datetime if not already present
    if 'datetime' not in df_fire.columns:
        raise ValueError("datetime column not found in filtered data")

    print(f"  Loaded {len(df_fire):,} filtered fire detections")

    # Verify x,y coordinates exist (should already be in parquet from embedding)
    if 'x' not in df_fire.columns or 'y' not in df_fire.columns:
        raise ValueError("Projected coordinates (x, y) not found in data")

    # Create sliding windows
    df_windows = create_sliding_windows(df_fire, window_hours=48, stride_hours=24)

    # Save outputs (overwrite existing embedded data)
    embedded_dir = repo_root / 'embedded_data'
    output_viirs = embedded_dir / 'nasa_viirs_with_weather.parquet'
    output_windows = embedded_dir / 'sliding_windows_index.parquet'
    output_params = embedded_dir / 'sliding_windows_params.json'

    # Save filtered fire data (keeps all existing columns from original parquet)
    df_fire.to_parquet(output_viirs, index=False)
    df_windows.to_parquet(output_windows, index=False)

    params = {
        'window_hours': 48,
        'stride_hours': 24,
        'num_windows': len(df_windows),
        'time_range': {
            'start': str(df_fire['datetime'].min()),
            'end': str(df_fire['datetime'].max())
        }
    }

    with open(output_params, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\n✓ Saved filtered fire data: {output_viirs}")
    print(f"✓ Saved sliding windows: {output_windows}")
    print(f"✓ Saved window parameters: {output_params}")
    print(f"\nReady for tiling pipeline!")


if __name__ == '__main__':
    main()
