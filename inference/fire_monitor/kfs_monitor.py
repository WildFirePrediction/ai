"""
KFS Fire Monitor
Continuously monitors KFS API for new fires and sends them to Flask inference server

MODES:
    - Production (--demo=False): Poll real KFS API, send to external backend
    - Demo (--demo=True): Generate fake fires, save locally with visualization

USAGE:
    # Production mode
    python inference/fire_monitor/kfs_monitor.py --poll-interval 300

    # Demo mode for presentations
    python inference/fire_monitor/kfs_monitor.py --demo --poll-interval 120
"""
import sys
from pathlib import Path
import time
import requests
import json
from datetime import datetime
import argparse
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from inference.fire_monitor.config import (
    KFS_REALTIME_URL,
    KFS_POLL_INTERVAL,
    FLASK_PREDICT_ENDPOINT,
    FLASK_HEALTH_ENDPOINT,
    PROCESSED_FIRES_DB
)

# Import fake fire generator from demo_rl
sys.path.insert(0, str(project_root / 'inference' / 'demo_rl' / 'src'))
from generate_dummy_fire import generate_random_fire


class KFSFireMonitor:
    """
    Monitors KFS API for active fires and sends them to inference server
    Supports both production (real KFS API) and demo (fake fires) modes
    Tracks fire lifecycle: NEW -> ACTIVE -> ENDED
    """

    def __init__(self, poll_interval=KFS_POLL_INTERVAL, demo_mode=False):
        """
        Initialize KFS monitor

        Args:
            poll_interval: Time between polls in seconds (default 120 = 2 min)
            demo_mode: If True, generate fake fires instead of polling KFS API
        """
        self.poll_interval = poll_interval
        self.demo_mode = demo_mode
        self.processed_fires = self._load_processed_fires()

        # Track active fires with their metadata
        # Format: {fire_id: {lat, lon, status, status_name, timestamp, last_seen_poll}}
        self.active_fires = {}

        # Demo mode: track when demo fires should end
        self.demo_fire_end_times = {}  # {fire_id: poll_count_when_to_end}

        # Status codes
        self.ACTIVE_STATUS_CODES = ['01', '02']  # 01=진행중, 02=진화중
        self.ENDED_STATUS_CODES = ['03']  # 03=진화완료 (Fire extinguished)

        # Create logs directory
        log_dir = Path("inference/fire_monitor/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        print("="*70)
        if self.demo_mode:
            print("KFS Fire Monitor - DEMO MODE")
        else:
            print("KFS Fire Monitor - PRODUCTION MODE")
        print("="*70)
        if self.demo_mode:
            print(f"Mode: DEMO (Generating fake fires)")
            print(f"Output: inference/demo_rl/output/")
        else:
            print(f"KFS API: {KFS_REALTIME_URL}")
            print(f"Flask Server: {FLASK_PREDICT_ENDPOINT}")
        print(f"Poll Interval: {poll_interval} seconds ({poll_interval/60:.1f} minutes)")
        print("="*70)

    def _load_processed_fires(self):
        """Load set of already processed fire IDs from disk"""
        db_path = Path(PROCESSED_FIRES_DB)
        if db_path.exists():
            try:
                with open(db_path, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_fire_ids', []))
            except Exception as e:
                print(f"Warning: Could not load processed fires DB: {e}")
                return set()
        return set()

    def _save_processed_fires(self):
        """Save processed fire IDs to disk"""
        db_path = Path(PROCESSED_FIRES_DB)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(db_path, 'w') as f:
                json.dump({
                    'processed_fire_ids': list(self.processed_fires),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save processed fires DB: {e}")

    def fetch_kfs_fires(self):
        """
        Fetch active fires from KFS API (production mode)
        OR generate fake fires (demo mode)

        Returns:
            list: List of fire dicts from KFS API or fake fires
        """
        if self.demo_mode:
            # Demo mode: generate fake fire
            print(f"  [DEMO] Generating fake fire...")
            fake_fire = generate_random_fire(use_known_zone=True)
            return [fake_fire]

        # Production mode: fetch from KFS API
        try:
            response = requests.get(KFS_REALTIME_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            fire_list = data.get("fireShowInfoList", [])
            return fire_list

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch KFS data: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse KFS response: {e}")
            return []

    def convert_kfs_to_inference_format(self, kfs_fire):
        """
        Convert KFS API format to inference server format

        KFS Format:
        {
            "frfrInfoId": "12345",
            "frfrLctnYcrd": "36.5684",  # Latitude
            "frfrLctnXcrd": "128.7294",  # Longitude
            "frfrFrngDtm": "2025-12-02 14:30:00",
            "frfrPrgrsStcd": "01",
            "frfrPrgrsStcdNm": "진행중"
        }

        Inference Format:
        {
            "fire_id": "12345",
            "latitude": 36.5684,
            "longitude": 128.7294,
            "timestamp": "2025-12-02T14:30:00"
        }

        Args:
            kfs_fire: Fire dict from KFS API

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

    def send_to_inference_server(self, fire_data):
        """
        Send fire data to Flask inference server

        Args:
            fire_data: Fire dict in inference format

        Returns:
            dict: Response from inference server or None if failed
        """
        try:
            # Add demo mode flag to request
            request_data = {**fire_data, 'demo_mode': self.demo_mode}

            response = requests.post(
                FLASK_PREDICT_ENDPOINT,
                json=request_data,
                timeout=120  # 2 minutes timeout for inference
            )
            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                print(f"  [SUCCESS] Prediction completed for fire {fire_data['fire_id']}")
                if self.demo_mode:
                    print(f"  [DEMO] Output saved to inference/demo_rl/output/")
                return result
            else:
                print(f"  [ERROR] Prediction failed: {result.get('error', 'Unknown error')}")
                return None

        except requests.exceptions.Timeout:
            print(f"  [ERROR] Inference server timeout for fire {fire_data['fire_id']}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Failed to send to inference server: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Failed to parse inference response: {e}")
            return None

    def send_fire_ended_notification(self, fire_id, fire_metadata, reason):
        """
        Send fire ended notification to Flask server (which forwards to backend)

        Args:
            fire_id: Fire unique identifier
            fire_metadata: Dict with fire location and other info
            reason: Reason for ending (disappeared, status_changed, demo_timeout)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            notification_data = {
                'event_type': '1',  # 0 = inference prediction, 1 = fire ended
                'fire_id': fire_id,
                'fire_location': {
                    'lat': fire_metadata.get('lat'),
                    'lon': fire_metadata.get('lon')
                },
                'fire_timestamp': fire_metadata.get('timestamp'),
                'ended_timestamp': datetime.now().isoformat(),
                'end_reason': reason,
                'last_status': fire_metadata.get('status_name', 'unknown'),
                'last_status_code': fire_metadata.get('status', 'unknown'),
                'demo_mode': self.demo_mode
            }

            # Add official completion time if available (from KFS potfrCmpleDtm field)
            if 'completion_timestamp' in fire_metadata:
                notification_data['completion_timestamp'] = fire_metadata['completion_timestamp']

            print(f"  [FIRE ENDED] {fire_id}")
            print(f"    Reason: {reason}")
            print(f"    Last location: {fire_metadata.get('lat'):.6f}°N, {fire_metadata.get('lon'):.6f}°E")
            print(f"    Last status: {fire_metadata.get('status_name')}")

            # Send to Flask server with fire_ended flag
            response = requests.post(
                FLASK_PREDICT_ENDPOINT,
                json=notification_data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                print(f"  [SUCCESS] Fire ended notification sent to backend")
                return True
            else:
                print(f"  [ERROR] Failed to send notification: {result.get('error', 'Unknown error')}")
                return False

        except requests.exceptions.Timeout:
            print(f"  [ERROR] Timeout sending fire ended notification for {fire_id}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Failed to send fire ended notification: {e}")
            return False
        except Exception as e:
            print(f"  [ERROR] Unexpected error sending notification: {e}")
            return False

    def check_flask_server_health(self):
        """
        Check if Flask inference server is running

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = requests.get(FLASK_HEALTH_ENDPOINT, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get('status') == 'healthy'
        except Exception:
            return False

    def process_fires(self, fire_list, current_poll):
        """
        Process list of fires from KFS API
        Tracks fire lifecycle: NEW -> ACTIVE -> ENDED

        Args:
            fire_list: List of fire dicts from KFS API
            current_poll: Current poll iteration number
        """
        if not fire_list:
            print(f"  No fires in current API response")

        # Step 1: Detect ended fires
        current_fire_ids = set([f.get("frfrInfoId", "unknown") for f in fire_list])
        ended_fires = []

        for fire_id, fire_metadata in list(self.active_fires.items()):
            ended = False
            reason = None

            # Check if fire disappeared from API
            if fire_id not in current_fire_ids:
                ended = True
                reason = "disappeared_from_api"

            # Check if status changed to ended code (for fires still in API)
            elif fire_id in current_fire_ids:
                current_fire = next((f for f in fire_list if f.get("frfrInfoId") == fire_id), None)
                if current_fire:
                    status_code = current_fire.get("frfrPrgrsStcd", "")
                    if status_code in self.ENDED_STATUS_CODES:
                        ended = True
                        reason = f"status_changed_to_{status_code}"
                        # Capture official completion time if available
                        completion_time = current_fire.get("potfrCmpleDtm", "")
                        if completion_time:
                            fire_metadata['completion_timestamp'] = completion_time.replace(" ", "T")

            # Demo mode: check if fire should end based on timer
            if self.demo_mode and fire_id in self.demo_fire_end_times:
                if current_poll >= self.demo_fire_end_times[fire_id]:
                    ended = True
                    reason = "demo_timeout"

            if ended:
                ended_fires.append((fire_id, fire_metadata, reason))

        # Send fire ended notifications
        for fire_id, fire_metadata, reason in ended_fires:
            self.send_fire_ended_notification(fire_id, fire_metadata, reason)
            # Remove from active fires
            del self.active_fires[fire_id]
            if fire_id in self.demo_fire_end_times:
                del self.demo_fire_end_times[fire_id]

        # Step 2: Process current fires
        print(f"  Found {len(fire_list)} fire(s) in current response")

        for kfs_fire in fire_list:
            fire_id = kfs_fire.get("frfrInfoId", "unknown")
            status_code = kfs_fire.get("frfrPrgrsStcd", "")
            status_name = kfs_fire.get("frfrPrgrsStcdNm", "Unknown")

            # Convert to inference format
            fire_data = self.convert_kfs_to_inference_format(kfs_fire)

            # Check if this is a new fire
            if fire_id not in self.processed_fires and fire_id not in self.active_fires:
                # Check if fire is already extinguished using frfrPrgrsStcd field
                if status_code in self.ENDED_STATUS_CODES:
                    print(f"\n  [SKIPPED] Fire {fire_id} already extinguished")
                    print(f"    Location: {fire_data['latitude']:.6f}°N, {fire_data['longitude']:.6f}°E")
                    print(f"    Timestamp: {fire_data['timestamp']}")
                    print(f"    Status: {status_name} (frfrPrgrsStcd={status_code})")
                    print(f"    Reason: Fire detected with completed status, no prediction needed")

                    # Mark as processed to avoid re-checking
                    self.processed_fires.add(fire_id)
                    self._save_processed_fires()
                    continue

                # NEW FIRE - send to inference
                print(f"\n  [NEW FIRE] {fire_id}")
                print(f"    Location: {fire_data['latitude']:.6f}°N, {fire_data['longitude']:.6f}°E")
                print(f"    Timestamp: {fire_data['timestamp']}")
                print(f"    Status: {fire_data['status']}")

                # Send to inference server
                result = self.send_to_inference_server(fire_data)

                if result:
                    # Mark as processed and active
                    self.processed_fires.add(fire_id)
                    self._save_processed_fires()

                    # Track as active fire
                    self.active_fires[fire_id] = {
                        'lat': fire_data['latitude'],
                        'lon': fire_data['longitude'],
                        'timestamp': fire_data['timestamp'],
                        'status': status_code,
                        'status_name': status_name,
                        'last_seen_poll': current_poll
                    }

                    # Demo mode: schedule fire ending
                    if self.demo_mode:
                        # End fire after 2-3 poll cycles
                        end_after = random.randint(2, 3)
                        self.demo_fire_end_times[fire_id] = current_poll + end_after
                        print(f"    [DEMO] Fire will end in {end_after} poll cycles")

                    # Log summary
                    num_predictions = len(result.get('predictions', []))
                    print(f"    Predictions: {num_predictions} timesteps generated")
                else:
                    print(f"    Failed to get predictions")

            elif fire_id in self.active_fires:
                # EXISTING FIRE - update metadata
                self.active_fires[fire_id].update({
                    'status': status_code,
                    'status_name': status_name,
                    'last_seen_poll': current_poll
                })
                print(f"  Fire {fire_id}: Still active, status={status_name}")

            else:
                # Fire was processed before but ended, skip
                pass

    def run(self):
        """
        Main monitoring loop
        Runs indefinitely until interrupted
        """
        mode_str = "DEMO MODE" if self.demo_mode else "PRODUCTION MODE"
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting KFS Fire Monitor ({mode_str})...")
        print("Press Ctrl+C to stop\n")

        # Initial health check
        if not self.check_flask_server_health():
            print("[WARNING] Flask inference server is not responding!")
            print(f"           Make sure server is running at {FLASK_PREDICT_ENDPOINT}")
            print("           Start with: python inference/rl/api_server.py --port 5000\n")

        iteration = 0
        while True:
            try:
                iteration += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{'='*70}")
                print(f"[{timestamp}] Poll #{iteration}")
                print(f"{'='*70}")

                # Fetch fires from KFS API
                fire_list = self.fetch_kfs_fires()

                # Process fires (pass current poll iteration)
                self.process_fires(fire_list, iteration)

                # Sleep until next poll
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sleeping for {self.poll_interval}s...")
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                print("\n\n[STOPPING] Monitor interrupted by user")
                print(f"Total fires processed: {len(self.processed_fires)}")
                break
            except Exception as e:
                print(f"\n[ERROR] Unexpected error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                print(f"Retrying in {self.poll_interval}s...")
                time.sleep(self.poll_interval)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='KFS Fire Monitor - Polls KFS API and triggers predictions'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=KFS_POLL_INTERVAL,
        help=f'Polling interval in seconds (default: {KFS_POLL_INTERVAL})'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo mode: generate fake fires instead of polling KFS API'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: fetch once and exit'
    )

    args = parser.parse_args()

    # Initialize monitor
    monitor = KFSFireMonitor(
        poll_interval=args.poll_interval,
        demo_mode=args.demo
    )

    if args.test:
        # Test mode: fetch once and exit
        print("\n[TEST MODE] Fetching KFS data once...\n")
        fire_list = monitor.fetch_kfs_fires()
        print(f"\nFetched {len(fire_list)} fires from KFS API")
        if fire_list:
            print("\nFire data:")
            print(json.dumps(fire_list, ensure_ascii=False, indent=2))
    else:
        # Normal mode: run indefinitely
        monitor.run()


if __name__ == '__main__':
    main()
