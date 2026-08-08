"""
Interactive web demo server for the wildfire spread prediction engine.

Serves the single page map UI and a small JSON API on top of the production
A3C inference pipeline. The model is loaded once into this process, so no
separate inference server is required.

ENDPOINTS:
    GET  /                 - Web demo UI
    GET  /api/health       - Engine status (used by the UI status indicator)
    GET  /api/presets      - Example ignition points for quick demos
    POST /api/coverage     - Validate a clicked point against data coverage
    POST /api/predict      - Run inference for one ignition point

USAGE:
    python webdemo/server.py --port 8080 --device cuda
    (or simply ./start_web.sh from the project root)
"""
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, send_from_directory  # noqa: E402
from flask_cors import CORS                                     # noqa: E402

from webdemo.engine import WebDemoEngine, TIMESTEPS             # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / 'static'

# Maximum rollout the UI is allowed to request (12 steps = 2 hours)
MAX_TIMESTEPS = 12

# Quick-start ignition points: notable Korean wildfire areas
PRESETS = [
    {'name': 'Uljin (Gyeongbuk)', 'lat': 36.9930, 'lon': 129.4004},
    {'name': 'Andong (Gyeongbuk)', 'lat': 36.5684, 'lon': 128.7294},
    {'name': 'Gangneung (Gangwon)', 'lat': 37.7519, 'lon': 128.8761},
    {'name': 'Sokcho (Gangwon)', 'lat': 38.2070, 'lon': 128.5918},
    {'name': 'Miryang (Gyeongnam)', 'lat': 35.4936, 'lon': 128.7368},
    {'name': 'Hongseong (Chungnam)', 'lat': 36.6010, 'lon': 126.6650},
]

app = Flask(__name__, static_folder=None)
CORS(app)

engine = None          # Populated by init_engine() before serve
fire_counter = 0       # Sequential id suffix for demo runs


# ----------------------------------------------------------------------
# Static assets
# ----------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the single page demo UI."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve CSS / JS assets."""
    return send_from_directory(STATIC_DIR, filename)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    """Engine status and model metadata."""
    if engine is None:
        return jsonify({
            'status': 'loading',
            'model_loaded': False,
            'timestamp': datetime.now().isoformat(),
        }), 503

    payload = {
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat(),
        'max_timesteps': MAX_TIMESTEPS,
    }
    payload.update(engine.info())
    return jsonify(payload)


@app.route('/api/presets', methods=['GET'])
def presets():
    """Example ignition points shown in the UI dropdown."""
    return jsonify({'presets': PRESETS})


@app.route('/api/coverage', methods=['POST'])
def coverage():
    """
    Check whether a clicked point can be simulated.

    Request:  {"lat": 36.5, "lon": 128.7}
    Response: {"ok": true} or {"ok": false, "message": "..."}
    """
    if engine is None:
        return jsonify({'ok': False, 'message': 'Engine still loading'}), 503

    data = request.get_json(silent=True) or {}
    try:
        lat = float(data['lat'])
        lon = float(data['lon'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'ok': False, 'message': 'lat and lon are required'}), 400

    ok, message = engine.check_coverage(lat, lon)
    return jsonify({'ok': ok, 'message': message})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run inference for a single ignition point.

    Request:
        {
            "lat": 36.5684,
            "lon": 128.7294,
            "timestamp": "2025-12-12T14:00:00",   # optional, defaults to now
            "timesteps": 3,                       # optional, 1..MAX_TIMESTEPS
            "fire_id": "WEB_0001"                 # optional
        }

    Response: full prediction payload (see webdemo/engine.py) plus
              {"success": true}
    """
    global fire_counter

    if engine is None:
        return jsonify({'success': False, 'error': 'Engine still loading'}), 503

    data = request.get_json(silent=True) or {}

    # Coordinates
    try:
        lat = float(data['lat'])
        lon = float(data['lon'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'error': 'lat and lon are required'}), 400

    ok, message = engine.check_coverage(lat, lon)
    if not ok:
        return jsonify({'success': False, 'error': message}), 400

    # Rollout horizon
    try:
        timesteps = int(data.get('timesteps', TIMESTEPS))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'timesteps must be an integer'}), 400
    timesteps = max(1, min(MAX_TIMESTEPS, timesteps))

    # Ignition time
    raw_timestamp = data.get('timestamp')
    if raw_timestamp:
        try:
            fire_timestamp = datetime.fromisoformat(str(raw_timestamp).replace('Z', '+00:00'))
            fire_timestamp = fire_timestamp.replace(tzinfo=None)
        except ValueError:
            return jsonify({'success': False,
                            'error': f'Invalid timestamp: {raw_timestamp}'}), 400
    else:
        fire_timestamp = datetime.now()

    fire_counter += 1
    fire_id = data.get('fire_id') or f"WEB{fire_counter:04d}"

    try:
        result = engine.predict(
            fire_id=fire_id,
            lat=lat,
            lon=lon,
            fire_timestamp=fire_timestamp,
            num_timesteps=timesteps,
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500

    result['success'] = True
    return jsonify(result)


# ----------------------------------------------------------------------
def init_engine(checkpoint, data_dir, output_dir, device):
    """Load the model into the module level engine handle."""
    global engine
    engine = WebDemoEngine(
        checkpoint_path=checkpoint,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Wildfire prediction web demo server')
    parser.add_argument('--checkpoint', type=str,
                        default='rl_training/a3c_16ch/V3_LSTM_REL/checkpoints/run1_relaxed/best_model.pt',
                        help='Path to the A3C model checkpoint')
    parser.add_argument('--data-dir', type=str, default='embedded_data',
                        help='Directory with static environmental rasters')
    parser.add_argument('--output-dir', type=str, default='webdemo/outputs',
                        help='Directory for saved prediction JSON files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Inference device (cuda or cpu)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Bind address (0.0.0.0 exposes the demo over Tailscale)')
    parser.add_argument('--port', type=int, default=8080, help='Bind port')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode')
    args = parser.parse_args()

    init_engine(args.checkpoint, args.data_dir, args.output_dir, args.device)

    print(f"\nWildfire Prediction Web Demo")
    print(f"{'=' * 70}")
    print(f"UI:        http://{args.host}:{args.port}/")
    print(f"Health:    http://{args.host}:{args.port}/api/health")
    print(f"Device:    {engine.device}")
    print(f"Outputs:   {args.output_dir}")
    print(f"{'=' * 70}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
