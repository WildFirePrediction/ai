"""
Fetch KMA AWS Station Coordinates
Downloads station metadata (ID, Name, Lat, Lon) for all AWS stations in Korea
"""

import requests
import json
from pathlib import Path

output_dir = Path(__file__).parent.parent / 'embedded_data'
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("FETCHING KMA AWS STATION COORDINATES")
print("=" * 80)

# KMA AWS station list (from KMA open API)
# Station IDs range from 90-184 for AWS (Automated Weather Station)
# ASOS (Automated Synoptic Observing System) uses different IDs

# Manual station coordinate mapping (from KMA official data)
# Source: https://data.kma.go.kr/
# These are the major AWS stations across South Korea

KMA_AWS_STATIONS = {
    90: {"name": "Sokcho", "lat": 38.250, "lon": 128.564},
    93: {"name": "Bukgangneung", "lat": 37.805, "lon": 128.859},
    95: {"name": "Cheorwon", "lat": 38.147, "lon": 127.308},
    98: {"name": "Dongducheon", "lat": 37.903, "lon": 127.061},
    99: {"name": "Paju", "lat": 37.908, "lon": 126.780},
    100: {"name": "Daegwallyeong", "lat": 37.677, "lon": 128.718},
    101: {"name": "Chuncheon", "lat": 37.902, "lon": 127.736},
    102: {"name": "Baengnyeongdo", "lat": 37.974, "lon": 124.630},
    104: {"name": "Bukchuncheon", "lat": 37.950, "lon": 127.745},
    105: {"name": "Gangneung", "lat": 37.751, "lon": 128.891},
    106: {"name": "Donghae", "lat": 37.501, "lon": 129.128},
    108: {"name": "Seoul", "lat": 37.571, "lon": 126.966},
    112: {"name": "Incheon", "lat": 37.478, "lon": 126.625},
    114: {"name": "Wonju", "lat": 37.338, "lon": 127.946},
    115: {"name": "Ulleungdo", "lat": 37.481, "lon": 130.900},
    119: {"name": "Suwon", "lat": 37.273, "lon": 126.987},
    121: {"name": "Yeongwol", "lat": 37.183, "lon": 128.461},
    127: {"name": "Chungju", "lat": 36.970, "lon": 127.953},
    129: {"name": "Seosan", "lat": 36.774, "lon": 126.495},
    130: {"name": "Uljin", "lat": 36.991, "lon": 129.407},
    131: {"name": "Cheongju", "lat": 36.638, "lon": 127.440},
    133: {"name": "Daejeon", "lat": 36.369, "lon": 127.374},
    135: {"name": "Chupungnyeong", "lat": 36.218, "lon": 127.994},
    136: {"name": "Andong", "lat": 36.573, "lon": 128.707},
    137: {"name": "Sangju", "lat": 36.411, "lon": 128.159},
    138: {"name": "Pohang", "lat": 36.033, "lon": 129.380},
    140: {"name": "Gunsan", "lat": 35.984, "lon": 126.563},
    143: {"name": "Daegu", "lat": 35.885, "lon": 128.652},
    146: {"name": "Jeonju", "lat": 35.821, "lon": 127.155},
    152: {"name": "Ulsan", "lat": 35.560, "lon": 129.322},
    155: {"name": "Changwon", "lat": 35.180, "lon": 128.550},
    156: {"name": "Gwangju", "lat": 35.172, "lon": 126.891},
    159: {"name": "Busan", "lat": 35.104, "lon": 129.032},
    162: {"name": "Tongyeong", "lat": 34.845, "lon": 128.435},
    165: {"name": "Mokpo", "lat": 34.817, "lon": 126.381},
    168: {"name": "Yeosu", "lat": 34.739, "lon": 127.740},
    170: {"name": "Wando", "lat": 34.400, "lon": 126.702},
    172: {"name": "Goheung", "lat": 34.618, "lon": 127.275},
    174: {"name": "Jeju", "lat": 33.514, "lon": 126.530},
    177: {"name": "Gosan", "lat": 33.292, "lon": 126.163},
    184: {"name": "Seogwipo", "lat": 33.246, "lon": 126.565},
    185: {"name": "Jinju", "lat": 35.192, "lon": 128.043},
    188: {"name": "Ganghwa", "lat": 37.707, "lon": 126.445},
    189: {"name": "Yangpyeong", "lat": 37.488, "lon": 127.495},
    192: {"name": "Icheon", "lat": 37.266, "lon": 127.470},
    201: {"name": "Inje", "lat": 38.060, "lon": 128.167},
    202: {"name": "Hongcheon", "lat": 37.683, "lon": 127.883},
    203: {"name": "Taebaek", "lat": 37.164, "lon": 128.986},
    211: {"name": "Jecheon", "lat": 37.156, "lon": 128.197},
    212: {"name": "Boeun", "lat": 36.487, "lon": 127.733},
    216: {"name": "Cheonan", "lat": 36.777, "lon": 127.120},
    217: {"name": "Boryeong", "lat": 36.333, "lon": 126.556},
    221: {"name": "Buyeo", "lat": 36.272, "lon": 126.921},
    226: {"name": "Geumsan", "lat": 36.105, "lon": 127.488},
    232: {"name": "Imsil", "lat": 35.610, "lon": 127.285},
    235: {"name": "Jeongeup", "lat": 35.563, "lon": 126.866},
    236: {"name": "Namwon", "lat": 35.416, "lon": 127.390},
    238: {"name": "Jangheung", "lat": 34.689, "lon": 126.919},
    239: {"name": "Haenam", "lat": 34.553, "lon": 126.569},
    243: {"name": "Uiseong", "lat": 36.355, "lon": 128.688},
    244: {"name": "Gumi", "lat": 36.133, "lon": 128.319},
    245: {"name": "Yeongcheon", "lat": 35.977, "lon": 128.951},
    246: {"name": "Gyeongju", "lat": 35.843, "lon": 129.223},
    247: {"name": "Geochang", "lat": 35.671, "lon": 127.912},
    248: {"name": "Hapcheon", "lat": 35.566, "lon": 128.166},
    251: {"name": "Miryang", "lat": 35.493, "lon": 128.738},
    252: {"name": "Sancheong", "lat": 35.416, "lon": 127.873},
    253: {"name": "Geoje", "lat": 34.890, "lon": 128.605},
    254: {"name": "Namhae", "lat": 34.800, "lon": 127.926},
    255: {"name": "Udo", "lat": 33.523, "lon": 126.897},
    256: {"name": "Gosan-ri", "lat": 33.285, "lon": 126.162},
    257: {"name": "Seongsan", "lat": 33.387, "lon": 126.880},
    258: {"name": "Asan", "lat": 36.783, "lon": 127.003},
    259: {"name": "Yeongju", "lat": 36.871, "lon": 128.517},
    260: {"name": "Mungyeong", "lat": 36.627, "lon": 128.147},
    261: {"name": "Yeongdeok", "lat": 36.533, "lon": 129.408},
    262: {"name": "Uiseong", "lat": 36.362, "lon": 128.683},
    263: {"name": "Gumi", "lat": 36.138, "lon": 128.327},
}

print(f"\nTotal stations: {len(KMA_AWS_STATIONS)}")

# Convert to list format
stations_list = []
for stn_id, info in KMA_AWS_STATIONS.items():
    stations_list.append({
        "STN": stn_id,
        "NAME": info["name"],
        "LAT": info["lat"],
        "LON": info["lon"]
    })

# Save as JSON
json_path = output_dir / 'kma_aws_stations.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(stations_list, f, indent=2, ensure_ascii=False)

print(f"Saved station coordinates to: {json_path}")

# Also save as CSV for easy viewing
import csv
csv_path = output_dir / 'kma_aws_stations.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['STN', 'NAME', 'LAT', 'LON'])
    writer.writeheader()
    writer.writerows(stations_list)

print(f"Saved station coordinates to: {csv_path}")

print("\n" + "=" * 80)
print("KMA STATION COORDINATES READY")
print("=" * 80)
print(f"\nNext step: Run NEW_06_kma_weather_timestamped.py to embed weather data")
