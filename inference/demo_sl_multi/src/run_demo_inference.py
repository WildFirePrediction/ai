"""
Demo SL inference script with multi-point initialization
Processes dummy fire JSON files with 3 initial burning points (1 center + 2 spread)
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import argparse
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from inference.sl.grid_utils import create_fire_grid, latlon_to_raster_crs
from inference.sl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_input_tensor
)
from inference.sl.inference_engine import WildfireInferenceEngine


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


def process_dummy_fire(fire_event, engine, static_loader, kma_url, output_dir):
    """
    Process a single multi-point dummy fire event through inference pipeline

    Args:
        fire_event: Fire event dict (KFS format with spread_points)
        engine: WildfireInferenceEngine instance
        static_loader: StaticDataLoader instance
        kma_url: KMA API URL
        output_dir: Directory to save predictions

    Returns:
        prediction_result: Dict with prediction results
        output_file: Path to saved file
    """
    fire_id = fire_event['frfrInfoId']
    lat = float(fire_event['frfrLctnYcrd'])
    lon = float(fire_event['frfrLctnXcrd'])
    timestamp_str = fire_event['frfrFrngDtm']
    spread_points = fire_event.get('spread_points', [])

    # Parse timestamp
    try:
        fire_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        print(f"Failed to parse timestamp: {timestamp_str}")
        return None, None

    print(f"\n{'=' * 80}")
    print(f"PROCESSING MULTI-POINT DUMMY FIRE EVENT")
    print(f"{'=' * 80}")
    print(f"Fire ID: {fire_id}")
    print(f"Center: ({lat:.4f}, {lon:.4f})")
    print(f"Spread points: {len(spread_points)}")
    for i, sp in enumerate(spread_points):
        print(f"  Point {i+1}: ({sp['lat']:.4f}, {sp['lon']:.4f})")
    print(f"Total initial cells: {1 + len(spread_points)}")
    print(f"Time: {fire_timestamp}")

    # Step 1: Create grid
    print(f"\n[1/6] Creating grid...")
    grid_bounds, grid_coords, center_xy = create_fire_grid(lat, lon)
    print(f"  Grid bounds: {grid_bounds[0]:.0f}m to {grid_bounds[1]:.0f}m")

    # Step 2: Extract static data
    print(f"\n[2/6] Extracting static data...")
    static_channels = static_loader.extract_static_features(grid_bounds)
    print(f"  Static channels: {static_channels.shape}")

    # Step 3: Fetch weather
    print(f"\n[3/6] Fetching weather data...")
    df_weather = fetch_kma_weather(fire_timestamp, kma_url)
    if df_weather is not None:
        print(f"  Found {len(df_weather)} weather stations")
    else:
        print(f"  WARNING: No weather data available")

    # Step 4: Process weather
    print(f"\n[4/6] Processing weather...")
    weather_channels = process_weather_data(df_weather, center_xy)
    print(f"  Weather channels: {weather_channels.shape}")

    # Step 5: Create multi-point fire mask
    print(f"\n[5/6] Creating multi-point fire mask...")

    # Convert spread points from lat/lon to raster CRS
    spread_points_xy = []
    for sp in spread_points:
        sp_x, sp_y = latlon_to_raster_crs(sp['lat'], sp['lon'])
        spread_points_xy.append((sp_x, sp_y))

    # Create fire mask with all points
    fire_mask = create_multi_point_fire_mask(
        center_xy,
        spread_points_xy,
        grid_coords,
        grid_size=30
    )
    fire_cells = (fire_mask > 0).sum()
    print(f"  Fire cells: {fire_cells}")

    # Step 6: Stack input and run inference
    print(f"\n[6/6] Running inference...")
    input_data = create_input_tensor(static_channels, weather_channels, fire_mask)
    print(f"  Input shape: {input_data.shape}")

    # Run model
    predictions = engine.predict(input_data)
    print(f"  Predictions shape: {predictions.shape}")

    # Process predictions to lat/lon
    results = engine.process_predictions(
        predictions, grid_coords, fire_timestamp
    )

    # Count predicted cells per timestep
    for r in results:
        print(f"  Timestep {r['timestep']}: {len(r['predicted_cells'])} cells predicted")

    # Create output JSON
    output_data = {
        'fire_id': fire_id,
        'fire_location': {
            'lat': lat,
            'lon': lon
        },
        'spread_points': spread_points,
        'total_initial_cells': int(fire_cells),
        'fire_timestamp': fire_timestamp.isoformat(),
        'inference_timestamp': datetime.now().isoformat(),
        'model': 'unet_16ch_v3',
        'predictions': results
    }

    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_safe = fire_timestamp.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fire_{fire_id}_{timestamp_safe}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nPrediction saved: {output_file}")
    print(f"{'=' * 80}\n")

    return output_data, output_file


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Run SL inference on multi-point dummy fire data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to multi-point dummy fire JSON file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='sl_training/unet_16ch_v3/checkpoints/run1_dilated(0.3642)/best.pt',
        help='Path to U-Net model checkpoint'
    )
    parser.add_argument(
        '--kma-url',
        type=str,
        default='https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?authKey=ud6z-X7yRDKes_l-8qQyFg',
        help='KMA API URL with auth key'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference/demo_sl_multi/output',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )

    args = parser.parse_args()

    # Load dummy fire data
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading multi-point dummy fire data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        kfs_response = json.load(f)

    fire_list = kfs_response.get('fireShowInfoList', [])
    if len(fire_list) == 0:
        print("Error: No fires found in input file")
        return

    print(f"Found {len(fire_list)} fire(s) to process")

    # Initialize components
    print("\n" + "=" * 80)
    print("INITIALIZING SL INFERENCE PIPELINE")
    print("=" * 80)

    print("\nLoading static data...")
    static_loader = StaticDataLoader()

    print("\nLoading U-Net inference engine...")
    engine = WildfireInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("=" * 80)

    # Process each fire
    output_files = []
    for i, fire_event in enumerate(fire_list):
        print(f"\n\nProcessing fire {i+1}/{len(fire_list)}...")
        try:
            result, output_file = process_dummy_fire(
                fire_event,
                engine,
                static_loader,
                args.kma_url,
                args.output_dir
            )
            if output_file:
                output_files.append(output_file)
        except Exception as e:
            print(f"Error processing fire: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Processed {len(output_files)}/{len(fire_list)} fires successfully")
    print(f"\nOutput files:")
    for f in output_files:
        print(f"  {f}")


if __name__ == '__main__':
    main()
