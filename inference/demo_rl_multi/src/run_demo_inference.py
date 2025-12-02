"""
Demo RL inference script with multi-point initialization
Processes dummy fire JSON files with 3 initial burning points (1 center + 2 spread)
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from inference.rl.inference_engine import WildfireRLInferenceEngine
from inference.rl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_rl_input_tensor
)
from inference.rl.grid_utils import (
    create_fire_grid,
    grid_cell_to_latlon
)

# Import KMA API URL from config
sys.path.insert(0, str(project_root / 'src'))
from config import KMA_AWS_BASE_URL


def create_multi_point_fire_mask(center_xy, spread_points_xy, grid_coords, grid_size=30):
    """
    Create initial fire mask with multiple burning cells (center + spread points)

    Args:
        center_xy: (center_x, center_y) center fire location in EPSG:5179
        spread_points_xy: List of (x, y) tuples for spread points in EPSG:5179
        grid_coords: (30, 30, 2) array of grid coordinates
        grid_size: Grid dimensions (default 30)

    Returns:
        fire_mask: (30, 30) binary mask with 1.0 at burning cells
    """
    fire_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Mark center cell
    center_x, center_y = center_xy
    distances = np.sqrt(
        (grid_coords[:, :, 0] - center_x)**2 +
        (grid_coords[:, :, 1] - center_y)**2
    )
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    fire_mask[min_idx] = 1.0

    # Mark spread point cells
    for spread_x, spread_y in spread_points_xy:
        distances = np.sqrt(
            (grid_coords[:, :, 0] - spread_x)**2 +
            (grid_coords[:, :, 1] - spread_y)**2
        )
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        fire_mask[min_idx] = 1.0

    return fire_mask


def run_demo_inference(input_json_path, output_dir, checkpoint_path, data_dir,
                      kma_api_url=KMA_AWS_BASE_URL, device='cuda'):
    """
    Run RL inference on a multi-point dummy fire JSON file

    Args:
        input_json_path: Path to dummy fire JSON with spread points
        output_dir: Directory to save prediction JSON
        checkpoint_path: Path to RL model checkpoint
        data_dir: Directory with static data
        kma_api_url: KMA API URL for weather data
        device: Device for inference

    Returns:
        output_path: Path to saved prediction JSON
    """
    # Import coordinate transformation
    from inference.rl.grid_utils import latlon_to_raster_crs

    # Load dummy fire data (KFS format with spread points)
    with open(input_json_path, 'r') as f:
        kfs_response = json.load(f)

    # Extract first fire from fireShowInfoList
    fire_event = kfs_response['fireShowInfoList'][0]
    fire_id = fire_event['frfrInfoId']
    fire_lat = float(fire_event['frfrLctnYcrd'])
    fire_lon = float(fire_event['frfrLctnXcrd'])
    fire_timestamp_str = fire_event['frfrFrngDtm']
    spread_points = fire_event.get('spread_points', [])

    print(f"="*70)
    print(f"RL Multi-Point Demo Inference - Fire {fire_id}")
    print(f"="*70)
    print(f"Center: {fire_lat:.6f}°N, {fire_lon:.6f}°E")
    print(f"Spread points: {len(spread_points)}")
    for i, sp in enumerate(spread_points):
        print(f"  Point {i+1}: {sp['lat']:.6f}°N, {sp['lon']:.6f}°E")
    print(f"Total burning cells: {1 + len(spread_points)}")
    print(f"Timestamp: {fire_timestamp_str}")

    # Parse timestamp
    try:
        fire_timestamp = datetime.strptime(fire_timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        print(f"Failed to parse timestamp: {fire_timestamp_str}")
        return None

    # Initialize inference engine
    print("\nInitializing RL inference engine...")
    engine = WildfireRLInferenceEngine(
        checkpoint_path=checkpoint_path,
        device=device,
        sequence_length=3
    )

    # Initialize static data loader
    print("Loading static data...")
    static_loader = StaticDataLoader(data_dir=data_dir)

    # Step 1: Create fire grid (centered at center point)
    print("\nStep 1: Creating fire grid...")
    grid_bounds, grid_coords, center_xy = create_fire_grid(
        fire_lat, fire_lon,
        grid_size=30,
        cell_size=400
    )
    print(f"  Grid bounds: {grid_bounds}")

    # Step 2: Extract static features
    print("\nStep 2: Extracting static features...")
    static_channels = static_loader.extract_static_features(grid_bounds)
    print(f"  Static channels: {static_channels.shape}")

    # Step 3: Fetch weather data from KMA API
    print("\nStep 3: Fetching real-time weather data from KMA...")
    df_weather = fetch_kma_weather(fire_timestamp, kma_api_url)

    if df_weather is None or len(df_weather) == 0:
        raise RuntimeError(f"Failed to fetch weather data from KMA API for timestamp {fire_timestamp}. Cannot proceed without weather data.")

    weather_channels = process_weather_data(df_weather, center_xy, grid_size=30)
    print(f"  Weather channels: {weather_channels.shape}")

    # Step 4: Create input tensor (16 channels for RL)
    print("\nStep 4: Creating input tensor...")
    env_data = create_rl_input_tensor(static_channels, weather_channels)
    print(f"  Input shape: {env_data.shape}")

    # Step 5: Create multi-point initial fire mask
    print("\nStep 5: Creating multi-point initial fire mask...")

    # Convert spread points from lat/lon to raster CRS
    spread_points_xy = []
    for sp in spread_points:
        sp_x, sp_y = latlon_to_raster_crs(sp['lat'], sp['lon'])
        spread_points_xy.append((sp_x, sp_y))

    # Create fire mask with all points
    initial_fire_mask = create_multi_point_fire_mask(
        center_xy,
        spread_points_xy,
        grid_coords,
        grid_size=30
    )
    print(f"  Fire mask: {initial_fire_mask.sum():.0f} burning cell(s)")

    # Step 6: Run iterative inference
    print("\nStep 6: Running RL inference (iterative)...")
    predictions = engine.predict_iterative(
        env_data=env_data,
        initial_fire_mask=initial_fire_mask,
        num_timesteps=5
    )

    # Step 7: Process predictions to lat/lon
    print("\nStep 7: Processing predictions...")
    timestep_minutes = 10  # 10 minutes per timestep
    results = engine.process_predictions(
        predictions=predictions,
        initial_fire_mask=initial_fire_mask,
        grid_coords=grid_coords,
        fire_timestamp=fire_timestamp,
        timestep_hours=timestep_minutes / 60.0
    )

    # Step 8: Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = fire_timestamp.strftime("%Y%m%d_%H%M%S")
    output_filename = f"fire_{fire_id}_{timestamp_str}.json"
    output_path = output_dir / output_filename

    output_data = {
        'fire_id': str(fire_id),
        'fire_location': {
            'lat': float(fire_lat),
            'lon': float(fire_lon)
        },
        'spread_points': spread_points,
        'total_initial_cells': int(initial_fire_mask.sum()),
        'fire_timestamp': fire_timestamp.isoformat(),
        'inference_timestamp': datetime.now().isoformat(),
        'model': 'a3c_16ch_v3_lstm_rel',
        'predictions': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Predictions saved to {output_path}")

    # Print summary
    total_predictions = sum(len(r['predicted_cells']) for r in results)
    print(f"\nSummary:")
    print(f"  Initial burning cells: {initial_fire_mask.sum():.0f}")
    print(f"  Total predicted cells: {total_predictions}")
    for r in results:
        print(f"  t+{r['timestep']}: {len(r['predicted_cells'])} cells")
    print(f"="*70)

    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RL Multi-Point Demo Inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to multi-point dummy fire JSON')
    parser.add_argument('--output-dir', type=str,
                       default='inference/demo_rl_multi/output',
                       help='Directory to save predictions')
    parser.add_argument('--checkpoint', type=str,
                       default='rl_training/a3c_16ch/V3_LSTM_REL/checkpoints/run1_relaxed/best_model.pt',
                       help='Path to RL model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='embedded_data',
                       help='Directory with static data')
    parser.add_argument('--kma-api', type=str, default=KMA_AWS_BASE_URL,
                       help='KMA API URL')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference')

    args = parser.parse_args()

    output_path = run_demo_inference(
        input_json_path=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        kma_api_url=args.kma_api,
        device=args.device
    )

    print(f"\n✓ Multi-point demo inference complete: {output_path}")
