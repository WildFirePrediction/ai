"""
Generate dummy fire data mimicking KFS API response
Creates random fire events for testing inference pipeline
"""
import json
import random
from datetime import datetime
from pathlib import Path
import argparse


# Gyeongsangbuk-do (North Gyeongsang Province) - Fire-prone region
# This province has the highest wildfire occurrence in South Korea
# Latitude: 36.0 to 37.5
# Longitude: 128.0 to 129.5
GYEONGBUK_LAT_MIN = 36.0
GYEONGBUK_LAT_MAX = 37.5
GYEONGBUK_LON_MIN = 128.0
GYEONGBUK_LON_MAX = 129.5

# Known fire-prone areas in Gyeongsangbuk-do
GYEONGBUK_FIRE_ZONES = [
    {'name': 'Andong', 'lat': 36.5684, 'lon': 128.7294, 'radius': 0.3},
    {'name': 'Pohang', 'lat': 36.0190, 'lon': 129.3435, 'radius': 0.3},
    {'name': 'Gyeongju', 'lat': 35.8562, 'lon': 129.2247, 'radius': 0.3},
    {'name': 'Gumi', 'lat': 36.1195, 'lon': 128.3446, 'radius': 0.3},
    {'name': 'Yeongju', 'lat': 36.8056, 'lon': 128.6236, 'radius': 0.3},
    {'name': 'Uljin', 'lat': 36.9930, 'lon': 129.4006, 'radius': 0.3},
    {'name': 'Yeongdeok', 'lat': 36.4150, 'lon': 129.3656, 'radius': 0.3},
]


def generate_random_fire(fire_id=None, use_known_zone=True):
    """
    Generate a single random fire event in Gyeongsangbuk-do

    Args:
        fire_id: Fire ID (auto-generated if None)
        use_known_zone: If True, use known fire-prone zones in Gyeongsangbuk-do

    Returns:
        fire_event: Dict mimicking KFS API response
    """
    if fire_id is None:
        fire_id = str(random.randint(400000, 500000))

    # Generate location in Gyeongsangbuk-do
    if use_known_zone and random.random() > 0.2:
        # Pick from known fire zones in Gyeongsangbuk-do (80% probability)
        zone = random.choice(GYEONGBUK_FIRE_ZONES)
        lat = zone['lat'] + random.uniform(-zone['radius'], zone['radius'])
        lon = zone['lon'] + random.uniform(-zone['radius'], zone['radius'])
        location_name = zone['name']
    else:
        # Random location in Gyeongsangbuk-do province
        lat = random.uniform(GYEONGBUK_LAT_MIN, GYEONGBUK_LAT_MAX)
        lon = random.uniform(GYEONGBUK_LON_MIN, GYEONGBUK_LON_MAX)
        location_name = 'Gyeongsangbuk-do'

    # Generate timestamp (current time or random recent time)
    timestamp = datetime.now()

    # Create fire event in KFS API format
    fire_event = {
        "frfrLctnXcrd": str(lon),  # Longitude (X coordinate)
        "frfrLctnYcrd": str(lat),  # Latitude (Y coordinate)
        "frfrFrngDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),  # Fire start time
        "frfrInfoId": fire_id,  # Fire ID
        "frfrPrgrsStcd": "01",  # Progress status (01=ongoing)
        "frfrPrgrsStcdNm": "진행중",  # Status name (ongoing)
        "frfrOccrrTpcd": "05",  # Occurrence type
        "frfrStepIssuCd": "00",  # Step code
        "frfrStepIssuNm": "초기 대응",  # Initial response
        "frfrPotfrRt": 100,  # Potential fire rate
        "frfrSttmnDt": timestamp.strftime("%Y%m%d"),  # Statement date
        "frfrSttmnHms": timestamp.strftime("%H%M%S"),  # Statement time
        "frfrOccrrStcd": "31",  # Occurrence status
        "frfrSttmnLctnXcrd": str(lon),  # Statement location X
        "frfrSttmnLctnYcrd": str(lat),  # Statement location Y
        "frfrSttmnAddr": f"경상북도 {location_name} (위도: {lat:.4f}, 경도: {lon:.4f})",  # Address
        "frfrSttmnAddrDe": f"경상북도 {location_name} 산림지역",  # Detailed address
        "potfrCmpleDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),  # Potential complete time
        "lgdngCd": str(random.randint(1000000, 9999999))  # Local government code
    }

    return fire_event


def generate_dummy_kfs_response(num_fires=1, use_known_zone=True):
    """
    Generate complete KFS API response with multiple fires in Gyeongsangbuk-do

    Args:
        num_fires: Number of fires to generate
        use_known_zone: Use known fire-prone zones in Gyeongsangbuk-do

    Returns:
        kfs_response: Dict mimicking KFS API response format
    """
    fire_list = []
    for i in range(num_fires):
        fire_event = generate_random_fire(use_known_zone=use_known_zone)
        fire_list.append(fire_event)

    kfs_response = {
        "fireShowInfoList": fire_list
    }

    return kfs_response


def save_dummy_fire(output_dir, num_fires=1, use_known_zone=True):
    """
    Generate and save dummy fire data to JSON file (Gyeongsangbuk-do only)

    Args:
        output_dir: Directory to save output
        num_fires: Number of fires to generate
        use_known_zone: Use known fire-prone zones in Gyeongsangbuk-do

    Returns:
        output_file: Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dummy data
    kfs_response = generate_dummy_kfs_response(num_fires, use_known_zone)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dummy_fire_{timestamp}.json"

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kfs_response, f, indent=2, ensure_ascii=False)

    print(f"Generated {num_fires} dummy fire(s)")
    print(f"Saved to: {output_file}")

    # Print fire details
    for i, fire in enumerate(kfs_response['fireShowInfoList']):
        lat = float(fire['frfrLctnYcrd'])
        lon = float(fire['frfrLctnXcrd'])
        fire_id = fire['frfrInfoId']
        timestamp = fire['frfrFrngDtm']
        print(f"\n  Fire {i+1}:")
        print(f"    ID: {fire_id}")
        print(f"    Location: ({lat:.4f}, {lon:.4f})")
        print(f"    Time: {timestamp}")

    return output_file


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate dummy fire data in Gyeongsangbuk-do for testing inference pipeline'
    )
    parser.add_argument(
        '--num-fires',
        type=int,
        default=1,
        help='Number of fires to generate (default: 1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference/demo/input',
        help='Output directory for dummy data'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random locations in Gyeongsangbuk-do instead of known fire zones'
    )

    args = parser.parse_args()

    # Generate and save
    save_dummy_fire(
        output_dir=args.output_dir,
        num_fires=args.num_fires,
        use_known_zone=not args.random
    )


if __name__ == '__main__':
    main()
