"""
Production inference loop for real-time wildfire prediction
Monitors KFS API and generates predictions for active fires
"""
import sys
from pathlib import Path
import time
import json
import requests
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from inference.sl.grid_utils import create_fire_grid, create_initial_fire_mask
from inference.sl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_input_tensor
)
from inference.sl.inference_engine import WildfireInferenceEngine


class ProductionInferenceLoop:
    """Main production inference loop"""

    def __init__(self, config):
        """
        Initialize production inference loop

        Args:
            config: Dictionary with configuration
                - checkpoint_path: Path to model checkpoint
                - kfs_url: KFS API URL
                - kma_url: KMA API URL
                - poll_interval: Seconds between API polls
                - output_dir: Directory to save predictions
                - device: Device for inference ('cuda' or 'cpu')
        """
        self.config = config

        # Initialize components
        print("=" * 80)
        print("WILDFIRE PREDICTION - PRODUCTION INFERENCE")
        print("=" * 80)

        # Load static data
        print("\nInitializing static data loader...")
        self.static_loader = StaticDataLoader(
            data_dir=config.get('data_dir', 'embedded_data')
        )

        # Load inference engine
        print("\nInitializing inference engine...")
        self.engine = WildfireInferenceEngine(
            checkpoint_path=config['checkpoint_path'],
            device=config.get('device', 'cuda')
        )

        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'inference/sl/outputs/production'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {self.output_dir}")

        # Track active fires
        self.active_fires = {}  # fire_id -> fire_info
        self.poll_interval = config.get('poll_interval', 60)

        print(f"Poll interval: {self.poll_interval}s")
        print("=" * 80)

    def poll_kfs_api(self):
        """
        Poll KFS API for active fires

        Returns:
            fire_list: List of fire event dicts
        """
        try:
            response = requests.get(self.config['kfs_url'], timeout=10)
            data = response.json()
            fire_list = data.get('fireShowInfoList', [])
            return fire_list
        except Exception as e:
            print(f"Error polling KFS API: {e}")
            return []

    def is_new_fire(self, fire_id):
        """Check if fire is new (not seen before)"""
        return fire_id not in self.active_fires

    def is_extinguished(self, fire_event):
        """Check if fire is extinguished (frfrPrgrsStcd == '03')"""
        return fire_event.get('frfrPrgrsStcd') == '03'

    def process_fire_event(self, fire_event):
        """
        Process a single fire event and generate prediction

        Args:
            fire_event: Fire event dict from KFS API

        Returns:
            prediction_result: Dict with prediction results
        """
        fire_id = fire_event['frfrInfoId']
        lat = float(fire_event['frfrLctnYcrd'])
        lon = float(fire_event['frfrLctnXcrd'])
        timestamp_str = fire_event['frfrFrngDtm']

        # Parse timestamp
        try:
            fire_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            print(f"Failed to parse timestamp: {timestamp_str}")
            return None

        print(f"\n{'=' * 80}")
        print(f"PROCESSING FIRE EVENT")
        print(f"{'=' * 80}")
        print(f"Fire ID: {fire_id}")
        print(f"Location: ({lat:.4f}, {lon:.4f})")
        print(f"Time: {fire_timestamp}")
        print(f"Status: {fire_event.get('frfrPrgrsStcdNm', 'Unknown')}")

        # Step 1: Create grid
        print(f"\n[1/6] Creating grid...")
        grid_bounds, grid_coords, center_xy = create_fire_grid(lat, lon)
        print(f"  Grid bounds: {grid_bounds[0]:.0f}m to {grid_bounds[1]:.0f}m")

        # Step 2: Extract static data
        print(f"\n[2/6] Extracting static data...")
        static_channels = self.static_loader.extract_static_features(grid_bounds)
        print(f"  Static channels: {static_channels.shape}")

        # Step 3: Fetch weather
        print(f"\n[3/6] Fetching weather data...")
        df_weather = fetch_kma_weather(fire_timestamp, self.config['kma_url'])
        if df_weather is not None:
            print(f"  Found {len(df_weather)} weather stations")
        else:
            print(f"  WARNING: No weather data available")

        # Step 4: Process weather
        print(f"\n[4/6] Processing weather...")
        weather_channels = process_weather_data(df_weather, center_xy)
        print(f"  Weather channels: {weather_channels.shape}")

        # Step 5: Create fire mask
        print(f"\n[5/6] Creating fire mask...")
        fire_mask = create_initial_fire_mask(center_xy, grid_coords)
        fire_cells = (fire_mask > 0).sum()
        print(f"  Fire cells: {fire_cells}")

        # Step 6: Stack input and run inference
        print(f"\n[6/6] Running inference...")
        input_data = create_input_tensor(static_channels, weather_channels, fire_mask)
        print(f"  Input shape: {input_data.shape}")

        # Run model
        predictions = self.engine.predict(input_data)
        print(f"  Predictions shape: {predictions.shape}")

        # Process predictions to lat/lon
        results = self.engine.process_predictions(
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
            'fire_timestamp': fire_timestamp.isoformat(),
            'inference_timestamp': datetime.now().isoformat(),
            'model': 'unet_16ch_v3',
            'predictions': results
        }

        # Save to file
        timestamp_safe = fire_timestamp.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"fire_{fire_id}_{timestamp_safe}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nPrediction saved: {output_file}")
        print(f"{'=' * 80}\n")

        return output_data

    def run(self):
        """Main monitoring loop"""
        print("\nStarting production inference loop...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Poll KFS API
                fire_list = self.poll_kfs_api()

                if len(fire_list) == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No active fires")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"Found {len(fire_list)} fire event(s)")

                # Process each fire
                for fire_event in fire_list:
                    fire_id = fire_event['frfrInfoId']

                    # Skip extinguished fires
                    if self.is_extinguished(fire_event):
                        if fire_id in self.active_fires:
                            print(f"Fire {fire_id} extinguished, removing from tracking")
                            del self.active_fires[fire_id]
                        continue

                    # Process only new fires
                    if self.is_new_fire(fire_id):
                        print(f"\nNEW FIRE DETECTED: {fire_id}")
                        try:
                            result = self.process_fire_event(fire_event)
                            if result:
                                self.active_fires[fire_id] = {
                                    'first_seen': datetime.now(),
                                    'fire_event': fire_event,
                                    'prediction': result
                                }
                        except Exception as e:
                            print(f"Error processing fire {fire_id}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"Fire {fire_id} already tracked, skipping")

                # Sleep until next poll
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n\nStopping inference loop...")
            print(f"Total fires processed: {len(self.active_fires)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Production inference for wildfire prediction'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='sl_training/unet_16ch_v3/checkpoints/run1_dilated(0.3642)/best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--kfs-url',
        type=str,
        default='https://fd.forest.go.kr/ffas/pubConn/selectPublicFireShowList.do',
        help='KFS API URL'
    )
    parser.add_argument(
        '--kma-url',
        type=str,
        default='https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?authKey=ud6z-X7yRDKes_l-8qQyFg',
        help='KMA API URL with auth key'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=60,
        help='Seconds between API polls (default: 60)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference/sl/outputs/production',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='embedded_data',
        help='Directory with static rasters'
    )

    args = parser.parse_args()

    # Build config
    config = {
        'checkpoint_path': args.checkpoint,
        'kfs_url': args.kfs_url,
        'kma_url': args.kma_url,
        'poll_interval': args.poll_interval,
        'output_dir': args.output_dir,
        'device': args.device,
        'data_dir': args.data_dir
    }

    # Run inference loop
    loop = ProductionInferenceLoop(config)
    loop.run()


if __name__ == '__main__':
    main()
