"""
Flask REST API Server for RL Wildfire Inference
Receives fire data, runs inference, and sends predictions to external backend

ENDPOINTS:
    GET  /health  - Health check
    POST /predict - Run inference and forward to external backend

USAGE:
    # Development
    python inference/rl/api_server.py --port 5000 --debug

    # Production
    gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 inference.rl.api_server:app
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import traceback

from inference.rl.inference_engine import WildfireRLInferenceEngine
from inference.rl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_rl_input_tensor
)
from inference.rl.grid_utils import (
    create_fire_grid,
    create_initial_fire_mask
)

# Import config
sys.path.insert(0, str(project_root / 'src'))
from config import KMA_AWS_BASE_URL

# Import fire monitor config for external backend URL
sys.path.insert(0, str(project_root / 'inference' / 'fire_monitor'))
from inference.fire_monitor.config import EXTERNAL_BACKEND_URL, EXTERNAL_BACKEND_TIMEOUT

# Import visualization module from demo_rl
sys.path.insert(0, str(project_root / 'inference' / 'demo_rl' / 'src'))
from visualize_prediction import create_prediction_map


# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global inference system (initialized on startup)
inference_system = None
inference_lock = threading.Lock()


class RLInferenceAPI:
    """API wrapper for RL inference system"""

    def __init__(self, checkpoint_path, data_dir, output_dir, device='cuda'):
        """
        Initialize inference API

        Args:
            checkpoint_path: Path to A3C model checkpoint
            data_dir: Directory with static raster data
            output_dir: Directory to save predictions
            device: Device for inference
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Initializing RL Inference API...")
        self.engine = WildfireRLInferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            sequence_length=3
        )
        self.static_loader = StaticDataLoader(data_dir=data_dir)
        print("RL Inference API ready")

    def predict(self, fire_data, kma_api_url=KMA_AWS_BASE_URL, demo_mode=False):
        """
        Run inference for a single fire

        Args:
            fire_data: Dict with keys:
                - fire_id: str
                - latitude: float
                - longitude: float
                - timestamp: str (ISO format)
            kma_api_url: KMA weather API URL
            demo_mode: If True, save to demo_rl/output and skip external backend

        Returns:
            prediction_result: Dict with prediction data
        """
        fire_id = fire_data['fire_id']
        fire_lat = float(fire_data['latitude'])
        fire_lon = float(fire_data['longitude'])
        fire_timestamp_str = fire_data['timestamp']

        # Parse timestamp
        if 'T' in fire_timestamp_str:
            fire_timestamp = datetime.fromisoformat(fire_timestamp_str.replace('Z', '+00:00'))
        else:
            fire_timestamp = datetime.strptime(fire_timestamp_str, "%Y-%m-%d %H:%M:%S")

        mode_str = "[DEMO]" if demo_mode else "[INFERENCE]"
        print(f"\n{mode_str} Processing fire {fire_id}")
        print(f"  Location: {fire_lat:.6f}N, {fire_lon:.6f}E")
        print(f"  Timestamp: {fire_timestamp.isoformat()}")

        # Run inference pipeline
        grid_bounds, grid_coords, center_xy = create_fire_grid(
            fire_lat, fire_lon, grid_size=30, cell_size=400
        )

        static_channels = self.static_loader.extract_static_features(grid_bounds)
        df_weather = fetch_kma_weather(fire_timestamp, kma_api_url)
        weather_channels = process_weather_data(df_weather, center_xy, grid_size=30)
        env_data = create_rl_input_tensor(static_channels, weather_channels)
        initial_fire_mask = create_initial_fire_mask(center_xy, grid_coords, grid_size=30)

        predictions = self.engine.predict_iterative(
            env_data=env_data,
            initial_fire_mask=initial_fire_mask,
            num_timesteps=5
        )

        results = self.engine.process_predictions(
            predictions=predictions,
            initial_fire_mask=initial_fire_mask,
            grid_coords=grid_coords,
            fire_timestamp=fire_timestamp,
            timestep_hours=10.0 / 60.0  # 10 minutes
        )

        # Determine output directory based on mode
        if demo_mode:
            output_dir = Path('inference/demo_rl/output')
        else:
            output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        timestamp_str = fire_timestamp.strftime("%Y%m%d_%H%M%S")
        output_filename = f"fire_{fire_id}_{timestamp_str}.json"
        output_path = output_dir / output_filename

        output_data = {
            'fire_id': str(fire_id),
            'fire_location': {
                'lat': float(fire_lat),
                'lon': float(fire_lon)
            },
            'fire_timestamp': fire_timestamp.isoformat(),
            'inference_timestamp': datetime.now().isoformat(),
            'model': 'a3c_16ch_v3_lstm_rel',
            'predictions': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  [SUCCESS] Saved predictions to {output_path}")

        # Create HTML visualization in demo mode
        if demo_mode:
            html_filename = f"fire_{fire_id}_{timestamp_str}_map.html"
            html_path = output_dir / html_filename
            create_prediction_map(output_data, html_path)
            print(f"  [DEMO] Created visualization: {html_path}")

        return output_data

    def send_to_external_backend(self, prediction_data, backend_url=EXTERNAL_BACKEND_URL):
        """
        Send prediction results to external production backend

        Args:
            prediction_data: Dict with prediction results
            backend_url: URL of external backend API

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"\n[EXTERNAL] Sending predictions to backend: {backend_url}")
            response = requests.post(
                backend_url,
                json=prediction_data,
                timeout=EXTERNAL_BACKEND_TIMEOUT
            )
            response.raise_for_status()

            print(f"  [SUCCESS] Backend received predictions (status {response.status_code})")
            return True

        except requests.exceptions.Timeout:
            print(f"  [ERROR] Backend timeout after {EXTERNAL_BACKEND_TIMEOUT}s")
            return False
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Failed to send to backend: {e}")
            return False


# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON: {'status': 'healthy', 'model': 'loaded', 'timestamp': '...'}
    """
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if inference_system else 'not loaded',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives fire data, runs inference, and optionally sends to external backend

    Request JSON:
        {
            "fire_id": "12345",
            "latitude": 36.5684,
            "longitude": 128.7294,
            "timestamp": "2025-11-30T14:30:00",
            "demo_mode": false  # Optional, default false
        }

    Response JSON:
        {
            "success": true,
            "fire_id": "12345",
            "predictions": [...],
            "sent_to_backend": true,
            "demo_mode": false
        }
    """
    try:
        if not inference_system:
            return jsonify({
                'success': False,
                'error': 'Inference system not initialized'
            }), 500

        # Get request data
        fire_data = request.get_json()

        # Check demo mode
        demo_mode = fire_data.pop('demo_mode', False)

        # Validate required fields
        required_fields = ['fire_id', 'latitude', 'longitude', 'timestamp']
        for field in required_fields:
            if field not in fire_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Run inference (thread-safe)
        with inference_lock:
            result = inference_system.predict(fire_data, demo_mode=demo_mode)

        # Send to external backend only in production mode
        backend_success = False
        if not demo_mode:
            backend_success = inference_system.send_to_external_backend(result)

        return jsonify({
            'success': True,
            'fire_id': result['fire_id'],
            'predictions': result['predictions'],
            'inference_timestamp': result['inference_timestamp'],
            'sent_to_backend': backend_success,
            'demo_mode': demo_mode,
            'message': 'Prediction completed successfully'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




def init_inference_system(checkpoint_path, data_dir, output_dir, device):
    """Initialize global inference system"""
    global inference_system
    inference_system = RLInferenceAPI(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device
    )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='RL Inference API Server')
    parser.add_argument('--checkpoint', type=str,
                       default='rl_training/a3c_16ch/V3_LSTM_REL/checkpoints/run1_relaxed/best_model.pt',
                       help='Path to RL model checkpoint')
    parser.add_argument('--data-dir', type=str, default='embedded_data',
                       help='Directory with static data')
    parser.add_argument('--output-dir', type=str, default='inference/rl/outputs/api',
                       help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')

    args = parser.parse_args()

    # Initialize inference system
    init_inference_system(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )

    # Start Flask server
    print(f"\nRL Inference API Server")
    print(f"{'='*70}")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"External Backend: {EXTERNAL_BACKEND_URL}")
    print(f"\nEndpoints:")
    print(f"  GET  /health   - Health check")
    print(f"  POST /predict  - Run inference and forward to backend")
    print(f"{'='*70}\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
