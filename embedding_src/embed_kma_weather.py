"""
KMA Weather Data Embedding - 9 Channels
Embeds weather data from KMA CSV files into 400m resolution grid
Channels: WD1, WS1, TA, HM, PA, RN-15m, RN-60m, RN-DAY, TD
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import sys

# Grid parameters (EPSG:5179 - Korean TM)
GRID_RESOLUTION = 400  # meters
GRID_SIZE = 3000  # 3000x3000 grid = 1200km x 1200km
X_MIN, Y_MIN = -200000, -500000
X_MAX = X_MIN + GRID_SIZE * GRID_RESOLUTION
Y_MAX = Y_MIN + GRID_SIZE * GRID_RESOLUTION

# 9 Weather channels
WEATHER_FEATURES = ['WD1', 'WS1', 'TA', 'HM', 'PA', 'RN-15m', 'RN-60m', 'RN-DAY', 'TD']

def load_station_coords():
    """Load AWS station coordinates (EPSG:5179)"""
    station_file = '/home/chaseungjoon/code/WildfirePrediction/data/KMA/kma_weather_station_5179.csv'
    df = pd.read_csv(station_file)
    coords = {}
    for _, row in df.iterrows():
        coords[int(row['STN'])] = (row['X'], row['Y'])
    return coords

def embed_weather_data(timestamp_dir, station_coords, output_dir):
    """Embed weather data for single timestamp"""
    timestamp = os.path.basename(timestamp_dir)
    csv_file = os.path.join(timestamp_dir, f'AWS_{timestamp}.csv')
    
    if not os.path.exists(csv_file):
        return False
    
    df = pd.read_csv(csv_file)
    
    # Filter valid stations and data
    valid_data = []
    for _, row in df.iterrows():
        stn = int(row['STN'])
        if stn not in station_coords:
            continue
        
        x, y = station_coords[stn]
        
        # Extract 9 features, handle missing values
        features = []
        for feat in WEATHER_FEATURES:
            val = row[feat]
            # Replace invalid values with NaN
            if val == -99.9 or val == -99.7 or pd.isna(val):
                val = np.nan
            features.append(val)
        
        # Skip if too many missing values
        if np.isnan(features).sum() > 5:
            continue
            
        valid_data.append([x, y] + features)
    
    if len(valid_data) < 10:
        return False
    
    # Convert to array
    data = np.array(valid_data)
    points = data[:, :2]  # x, y coordinates
    values = data[:, 2:]  # 9 features
    
    # Create grid
    x_grid = np.arange(X_MIN, X_MAX, GRID_RESOLUTION)
    y_grid = np.arange(Y_MIN, Y_MAX, GRID_RESOLUTION)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Interpolate each channel
    embedded = np.zeros((GRID_SIZE, GRID_SIZE, 9), dtype=np.float32)
    
    for i in range(9):
        channel_values = values[:, i]
        # Remove NaN for interpolation
        mask = ~np.isnan(channel_values)
        if mask.sum() < 3:
            continue
        
        interpolated = griddata(
            points[mask], 
            channel_values[mask],
            grid_points,
            method='linear',
            fill_value=np.nanmean(channel_values[mask])
        )
        embedded[:, :, i] = interpolated.reshape(GRID_SIZE, GRID_SIZE)
    
    # Save
    output_file = os.path.join(output_dir, f'{timestamp}_weather.npy')
    np.save(output_file, embedded)
    return True

def main():
    print("=" * 80)
    print("KMA Weather Data Embedding Pipeline - 9 Channels")
    print("=" * 80)
    
    kma_dir = '/home/chaseungjoon/code/WildfirePrediction/data/KMA'
    output_dir = '/home/chaseungjoon/code/WildfirePrediction/embedded_data/kma_weather'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading station coordinates...")
    station_coords = load_station_coords()
    print(f"Loaded {len(station_coords)} weather stations")
    
    # Get all timestamp directories
    timestamp_dirs = sorted([d for d in Path(kma_dir).iterdir() if d.is_dir()])
    total = len(timestamp_dirs)
    
    print(f"\nProcessing {total} timestamps...")
    print("=" * 80)
    
    success_count = 0
    for idx, ts_dir in enumerate(timestamp_dirs, 1):
        success = embed_weather_data(str(ts_dir), station_coords, output_dir)
        if success:
            success_count += 1
        
        # Print progress every 100 timestamps
        if idx % 100 == 0 or idx == total:
            progress = (idx / total) * 100
            print(f"Progress: {idx}/{total} ({progress:.1f}%) | Success: {success_count} | Current: {ts_dir.name}")
            sys.stdout.flush()
    
    print("=" * 80)
    print(f"\nEmbedding Complete!")
    print(f"Total processed: {total}")
    print(f"Successfully embedded: {success_count}")
    print(f"Success rate: {success_count/total*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
