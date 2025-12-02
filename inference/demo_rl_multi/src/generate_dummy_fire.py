"""
Generate dummy fire data with multiple initial points (simulating spread)
Creates 1 initial point + 2 adjacent points (3 total) representing t=0 and t=1
"""
import json
import random
from datetime import datetime
from pathlib import Path
import argparse


# Gyeongsangbuk-do (North Gyeongsang Province) - Fire-prone region
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


def generate_multi_point_fire(fire_id=None, use_known_zone=True):
    """
    Generate a fire event with multiple initial points (1 center + 2 adjacent)
    Simulates that fire has already spread one timestep from the initial point

    Args:
        fire_id: Fire ID (auto-generated if None)
        use_known_zone: If True, use known fire-prone zones in Gyeongsangbuk-do

    Returns:
        fire_event: Dict with center point and 2 spread points
    """
    if fire_id is None:
        fire_id = str(random.randint(400000, 500000))

    # Generate center location in Gyeongsangbuk-do
    if use_known_zone and random.random() > 0.2:
        # Pick from known fire zones (80% probability)
        zone = random.choice(GYEONGBUK_FIRE_ZONES)
        center_lat = zone['lat'] + random.uniform(-zone['radius'], zone['radius'])
        center_lon = zone['lon'] + random.uniform(-zone['radius'], zone['radius'])
        location_name = zone['name']
    else:
        # Random location in Gyeongsangbuk-do province
        center_lat = random.uniform(GYEONGBUK_LAT_MIN, GYEONGBUK_LAT_MAX)
        center_lon = random.uniform(GYEONGBUK_LON_MIN, GYEONGBUK_LON_MAX)
        location_name = 'Gyeongsangbuk-do'

    # Generate timestamp
    timestamp = datetime.now()

    # Generate 2 adjacent spread points (simulating one timestep of spread)
    # Cell size is 400m, so approximate offset in degrees
    # At lat ~36, 1 degree lon ≈ 89km, 1 degree lat ≈ 111km
    # 400m offset ≈ 0.0036 degrees
    cell_offset_deg = 0.0036

    # Pick 2 random adjacent directions (8-connected)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    spread_directions = random.sample(directions, 2)

    spread_points = []
    for dy, dx in spread_directions:
        spread_lat = center_lat + dy * cell_offset_deg
        spread_lon = center_lon + dx * cell_offset_deg
        spread_points.append({
            'lat': spread_lat,
            'lon': spread_lon
        })

    # Create fire event with multiple points
    fire_event = {
        "frfrLctnXcrd": str(center_lon),  # Center longitude
        "frfrLctnYcrd": str(center_lat),  # Center latitude
        "frfrFrngDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "frfrInfoId": fire_id,
        "frfrPrgrsStcd": "01",
        "frfrPrgrsStcdNm": "진행중",
        "frfrOccrrTpcd": "05",
        "frfrStepIssuCd": "00",
        "frfrStepIssuNm": "초기 대응",
        "frfrPotfrRt": 100,
        "frfrSttmnDt": timestamp.strftime("%Y%m%d"),
        "frfrSttmnHms": timestamp.strftime("%H%M%S"),
        "frfrOccrrStcd": "31",
        "frfrSttmnLctnXcrd": str(center_lon),
        "frfrSttmnLctnYcrd": str(center_lat),
        "frfrSttmnAddr": f"경상북도 {location_name} (위도: {center_lat:.4f}, 경도: {center_lon:.4f})",
        "frfrSttmnAddrDe": f"경상북도 {location_name} 산림지역",
        "potfrCmpleDtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "lgdngCd": str(random.randint(1000000, 9999999)),
        # Add spread points for multi-point initialization
        "spread_points": spread_points
    }

    return fire_event


def generate_dummy_kfs_response(num_fires=1, use_known_zone=True):
    """
    Generate KFS API response with multi-point fires

    Args:
        num_fires: Number of fires to generate
        use_known_zone: Use known fire-prone zones

    Returns:
        kfs_response: Dict mimicking KFS API response format with spread points
    """
    fire_list = []
    for i in range(num_fires):
        fire_event = generate_multi_point_fire(use_known_zone=use_known_zone)
        fire_list.append(fire_event)

    kfs_response = {
        "fireShowInfoList": fire_list
    }

    return kfs_response


def save_dummy_fire(output_dir, num_fires=1, use_known_zone=True):
    """
    Generate and save multi-point dummy fire data to JSON file

    Args:
        output_dir: Directory to save output
        num_fires: Number of fires to generate
        use_known_zone: Use known fire-prone zones

    Returns:
        output_file: Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dummy data
    kfs_response = generate_dummy_kfs_response(num_fires, use_known_zone)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dummy_fire_multi_{timestamp}.json"

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kfs_response, f, indent=2, ensure_ascii=False)

    print(f"Generated {num_fires} multi-point dummy fire(s)")
    print(f"Saved to: {output_file}")

    # Print fire details
    for i, fire in enumerate(kfs_response['fireShowInfoList']):
        center_lat = float(fire['frfrLctnYcrd'])
        center_lon = float(fire['frfrLctnXcrd'])
        fire_id = fire['frfrInfoId']
        timestamp_str = fire['frfrFrngDtm']
        spread_points = fire['spread_points']

        print(f"\n  Fire {i+1}:")
        print(f"    ID: {fire_id}")
        print(f"    Center: ({center_lat:.4f}, {center_lon:.4f})")
        print(f"    Spread points (2):")
        for j, sp in enumerate(spread_points):
            print(f"      Point {j+1}: ({sp['lat']:.4f}, {sp['lon']:.4f})")
        print(f"    Time: {timestamp_str}")
        print(f"    Total burning cells: 3 (1 center + 2 spread)")

    return output_file


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate multi-point dummy fire data in Gyeongsangbuk-do'
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
        default='inference/demo_rl_multi/input',
        help='Output directory for dummy data'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random locations instead of known fire zones'
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
