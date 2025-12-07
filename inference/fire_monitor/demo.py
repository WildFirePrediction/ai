"""
Demo Fire Injection Tool
Feeds fake fire data to running Flask server for end-to-end testing

This tool tests the FULL pipeline:
  Fire Data -> Flask Server -> Inference -> External Backend -> Frontend

USAGE:
    # Generate random fire in known zone
    python inference/fire_monitor/demo.py

    # Custom location
    python inference/fire_monitor/demo.py --lat 36.5684 --lon 128.7294

    # Custom fire ID
    python inference/fire_monitor/demo.py --fire-id TEST_001

    # Test fire ended notification
    python inference/fire_monitor/demo.py --ended --fire-id 12345

PREREQUISITES:
    - start_monitoring.sh must be running (Flask server on port 5000)
    - External backend must be configured in .env (EXTERNAL_BACKEND_URL)
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import requests

from generate_dummy_fire import GYEONGBUK_FIRE_ZONES, generate_random_fire
from inference.fire_monitor.config import (
    EXTERNAL_BACKEND_URLS,
    FLASK_HEALTH_ENDPOINT,
    FLASK_PREDICT_ENDPOINT,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# Import fake fire generator
sys.path.insert(0, str(project_root / 'inference' / 'demo_rl' / 'src'))


def convert_kfs_to_inference_format(kfs_fire):
    """
    Convert KFS API format to inference server format

    Args:
        kfs_fire: Fire dict from KFS API format

    Returns:
        dict: Fire data in inference format
    """
    return {
        "fire_id": kfs_fire.get("frfrInfoId", "unknown"),
        "latitude": float(kfs_fire.get("frfrLctnYcrd", 0)),
        "longitude": float(kfs_fire.get("frfrLctnXcrd", 0)),
        "timestamp": kfs_fire.get("frfrFrngDtm", datetime.now().isoformat()).replace(" ", "T"),
        "status": kfs_fire.get("frfrPrgrsStcdNm", "Unknown"),
        "status_code": kfs_fire.get("frfrPrgrsStcd", "")
    }


def generate_custom_fire(fire_id=None, lat=None, lon=None, use_known_zone=True):
    """
    Generate fire with custom parameters

    Args:
        fire_id: Custom fire ID (auto-generated if None)
        lat: Custom latitude (random in known zone if None)
        lon: Custom longitude (random in known zone if None)
        use_known_zone: Use known fire-prone zones if lat/lon not specified

    Returns:
        dict: Fire data in KFS API format
    """
    # Generate base fire
    if lat is None or lon is None:
        # Use generator to pick location
        fire = generate_random_fire(
            fire_id=fire_id, use_known_zone=use_known_zone)
    else:
        # Use custom location
        if fire_id is None:
            fire_id = f"DEMO_{random.randint(100000, 999999)}"

        timestamp = datetime.now()
        fire = {
            "frfrLctnXcrd": str(lon),
            "frfrLctnYcrd": str(lat),
            "frfrFrngDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "frfrInfoId": fire_id,
            "frfrPrgrsStcd": "02",
            "frfrPrgrsStcdNm": "진행중",
            "frfrOccrrTpcd": "05",
            "frfrStepIssuCd": "00",
            "frfrStepIssuNm": "초기 대응",
            "frfrPotfrRt": 100,
            "frfrSttmnDt": timestamp.strftime("%Y%m%d"),
            "frfrSttmnHms": timestamp.strftime("%H%M%S"),
            "frfrOccrrStcd": "31",
            "frfrSttmnLctnXcrd": str(lon),
            "frfrSttmnLctnYcrd": str(lat),
            "frfrSttmnAddr": f"Custom Location (Lat: {lat:.4f}, Lon: {lon:.4f})",
            "frfrSttmnAddrDe": f"Demo fire at {lat:.4f}N, {lon:.4f}E",
            "potfrCmpleDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "lgdngCd": str(random.randint(1000000, 9999999))
        }

    return fire


def generate_fire_ended_notification(fire_id, lat=None, lon=None, reason="demo_manual"):
    """
    Generate fire ended notification

    Args:
        fire_id: Fire ID that ended
        lat: Fire latitude (random if None)
        lon: Fire longitude (random if None)
        reason: End reason

    Returns:
        dict: Fire ended notification in inference format
    """
    if lat is None or lon is None:
        # Generate random location
        zone = random.choice(GYEONGBUK_FIRE_ZONES)
        lat = zone['lat'] + random.uniform(-zone['radius'], zone['radius'])
        lon = zone['lon'] + random.uniform(-zone['radius'], zone['radius'])

    timestamp = datetime.now()

    return {
        'event_type': '1',
        'fire_id': fire_id,
        'fire_location': {
            'lat': lat,
            'lon': lon
        },
        'fire_timestamp': (timestamp).isoformat(),
        'ended_timestamp': timestamp.isoformat(),
        'end_reason': reason,
        'last_status': '진행중',
        'last_status_code': '01',
        'demo_mode': False
    }


def check_server_health():
    """
    Check if Flask server is running

    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        response = requests.get(FLASK_HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"[ERROR] Flask server not responding: {e}")
        return False


def send_fire_to_server(fire_data):
    """
    Send fire data directly to Flask server

    Args:
        fire_data: Fire dict in inference format

    Returns:
        dict: Response from server or None if failed
    """
    try:
        print(f"\n[SENDING] Posting to {FLASK_PREDICT_ENDPOINT}")
        print(f"  Payload: {json.dumps(fire_data, indent=2)}")

        response = requests.post(
            FLASK_PREDICT_ENDPOINT,
            json=fire_data,
            timeout=120  # 2 minutes for inference
        )
        response.raise_for_status()
        result = response.json()

        print(f"\n[RESPONSE] Status: {response.status_code}")
        print(json.dumps(result, indent=2))

        return result

    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Request timeout (120s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"  Response: {e.response.text}")
            except:
                pass
        return None
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse response: {e}")
        return None


def print_fire_summary(fire_data, event_type='prediction'):
    """Print summary of fire data being sent"""
    print("="*70)
    if event_type == 'prediction':
        print("DEMO FIRE INJECTION - NEW FIRE PREDICTION")
        print("="*70)
        print(f"Fire ID:    {fire_data.get('fire_id')}")
        print(
            f"Location:   {fire_data.get('latitude'):.6f}N, {fire_data.get('longitude'):.6f}E")
        print(f"Timestamp:  {fire_data.get('timestamp')}")
        print(f"Status:     {fire_data.get('status', 'N/A')}")
    else:
        print("DEMO FIRE INJECTION - FIRE ENDED NOTIFICATION")
        print("="*70)
        print(f"Fire ID:    {fire_data.get('fire_id')}")
        loc = fire_data.get('fire_location', {})
        print(f"Location:   {loc.get('lat'):.6f}N, {loc.get('lon'):.6f}E")
        print(f"Started:    {fire_data.get('fire_timestamp')}")
        print(f"Ended:      {fire_data.get('ended_timestamp')}")
        print(f"Reason:     {fire_data.get('end_reason')}")
    print("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Demo Fire Injection Tool - Test full pipeline with fake fires',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random fire in known zone
  python inference/fire_monitor/demo.py

  # Custom location
  python inference/fire_monitor/demo.py --lat 36.5684 --lon 128.7294

  # Custom fire ID
  python inference/fire_monitor/demo.py --fire-id TEST_001 --lat 36.5 --lon 128.7

  # Test fire ended notification
  python inference/fire_monitor/demo.py --ended --fire-id EXISTING_FIRE_123
        """
    )

    parser.add_argument(
        '--fire-id',
        type=str,
        help='Custom fire ID (auto-generated if not specified)'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help='Fire latitude (random in known zone if not specified)'
    )
    parser.add_argument(
        '--lon',
        type=float,
        help='Fire longitude (random in known zone if not specified)'
    )
    parser.add_argument(
        '--ended',
        action='store_true',
        help='Send fire ended notification instead of prediction request'
    )
    parser.add_argument(
        '--reason',
        type=str,
        default='demo_manual',
        help='End reason (only for --ended mode)'
    )
    parser.add_argument(
        '--no-known-zone',
        action='store_true',
        help='Generate random location anywhere (not just known zones)'
    )

    args = parser.parse_args()

    # Check if Flask server is running
    print("\n[CHECK] Verifying Flask server health...")
    if not check_server_health():
        print("\n[FAILED] Flask server is not responding!")
        print(f"  Endpoint: {FLASK_HEALTH_ENDPOINT}")
        print("\nPlease ensure the monitoring system is running:")
        print("  ./start_monitoring.sh")
        print("\nOr start Flask server manually:")
        print("  python inference/rl/api_server.py --port 5000")
        return 1

    print("[OK] Flask server is healthy")

    # Show backend configuration
    if EXTERNAL_BACKEND_URLS:
        print(
            f"\n[INFO] External backends configured: {len(EXTERNAL_BACKEND_URLS)}")
        for i, url in enumerate(EXTERNAL_BACKEND_URLS, 1):
            print(f"  {i}. {url}")
    else:
        print("\n[WARNING] No external backend configured in .env")
        print("  Predictions will be saved locally but not forwarded")

    # Generate or prepare fire data
    if args.ended:
        # Fire ended notification
        if not args.fire_id:
            print("\n[ERROR] --fire-id is required for fire ended notifications")
            return 1

        fire_data = generate_fire_ended_notification(
            fire_id=args.fire_id,
            lat=args.lat,
            lon=args.lon,
            reason=args.reason
        )
        event_type = 'ended'
    else:
        # New fire prediction
        kfs_fire = generate_custom_fire(
            fire_id=args.fire_id,
            lat=args.lat,
            lon=args.lon,
            use_known_zone=not args.no_known_zone
        )
        fire_data = convert_kfs_to_inference_format(kfs_fire)
        fire_data['demo_mode'] = False  # False to test full pipeline
        event_type = 'prediction'

    # Print summary
    print_fire_summary(fire_data, event_type)

    # Send to server
    result = send_fire_to_server(fire_data)

    if result:
        if result.get('success'):
            print("\n" + "="*70)
            print("[SUCCESS] Demo fire processed successfully!")
            print("="*70)

            if event_type == 'prediction':
                num_predictions = len(result.get('predictions', []))
                backend_success = result.get('backend_success_count', 0)
                backend_total = result.get('backend_total_count', 0)

                print(f"\nInference Results:")
                print(
                    f"  - Predictions generated: {num_predictions} timesteps")
                print(
                    f"  - Inference timestamp: {result.get('inference_timestamp')}")
                print(
                    f"  - Backend delivery: {backend_success}/{backend_total} successful")

                if backend_success > 0:
                    print(f"\n[OK] Data sent to external backend(s)")
                    print(f"  Check frontend to see the prediction visualization")
                else:
                    print(f"\n[WARNING] Could not reach external backend")
                    print(f"  Check if backend server is running")
                    print(f"  Predictions saved locally in: inference/rl/outputs/api/")
            else:
                print(f"\nFire Ended Notification:")
                backend_success = result.get('backend_success_count', 0)
                backend_total = result.get('backend_total_count', 0)
                print(
                    f"  - Backend delivery: {backend_success}/{backend_total} successful")

                if backend_success > 0:
                    print(f"\n[OK] Notification sent to external backend(s)")
                else:
                    print(f"\n[WARNING] Could not reach external backend")

            return 0
        else:
            print("\n" + "="*70)
            print("[FAILED] Server returned error")
            print("="*70)
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    else:
        print("\n" + "="*70)
        print("[FAILED] Could not communicate with server")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
