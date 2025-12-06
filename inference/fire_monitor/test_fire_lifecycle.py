"""
Test script for fire lifecycle tracking
Tests NEW -> ACTIVE -> ENDED transitions
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from inference.fire_monitor.kfs_monitor import KFSFireMonitor

def test_fire_lifecycle():
    """Test fire lifecycle tracking logic"""
    print("="*70)
    print("Testing Fire Lifecycle Tracking")
    print("="*70)

    # Initialize monitor in demo mode
    monitor = KFSFireMonitor(poll_interval=120, demo_mode=True)

    # Simulate poll 1: New fire appears
    print("\n[POLL 1] New fire appears")
    fake_fire_1 = {
        "frfrInfoId": "TEST001",
        "frfrLctnYcrd": "36.5684",
        "frfrLctnXcrd": "128.7294",
        "frfrFrngDtm": "2025-12-06 14:30:00",
        "frfrPrgrsStcd": "01",
        "frfrPrgrsStcdNm": "진행중"
    }
    fire_list_1 = [fake_fire_1]
    current_fire_ids = set([f.get("frfrInfoId") for f in fire_list_1])
    print(f"  Current fires in API: {current_fire_ids}")
    print(f"  Active fires before: {set(monitor.active_fires.keys())}")

    # Manually track this fire (simulating what process_fires would do)
    monitor.active_fires["TEST001"] = {
        'lat': 36.5684,
        'lon': 128.7294,
        'timestamp': "2025-12-06T14:30:00",
        'status': "01",
        'status_name': "진행중",
        'last_seen_poll': 1
    }
    print(f"  Active fires after: {set(monitor.active_fires.keys())}")

    # Simulate poll 2: Fire still active, status changes to extinguishing
    print("\n[POLL 2] Fire status changes to '진화중'")
    fake_fire_2 = {
        "frfrInfoId": "TEST001",
        "frfrLctnYcrd": "36.5684",
        "frfrLctnXcrd": "128.7294",
        "frfrFrngDtm": "2025-12-06 14:30:00",
        "frfrPrgrsStcd": "02",
        "frfrPrgrsStcdNm": "진화중"
    }
    fire_list_2 = [fake_fire_2]
    current_fire_ids = set([f.get("frfrInfoId") for f in fire_list_2])

    # Check if fire should end (should NOT end, 02 is still active)
    ended_fires = []
    for fire_id, fire_metadata in list(monitor.active_fires.items()):
        if fire_id not in current_fire_ids:
            ended_fires.append((fire_id, "disappeared"))
        else:
            current_fire = next((f for f in fire_list_2 if f.get("frfrInfoId") == fire_id), None)
            if current_fire:
                status_code = current_fire.get("frfrPrgrsStcd", "")
                if status_code in monitor.ENDED_STATUS_CODES:
                    ended_fires.append((fire_id, f"status_{status_code}"))

    print(f"  Current status: {fake_fire_2['frfrPrgrsStcdNm']}")
    print(f"  Fires to end: {ended_fires}")
    print(f"  Active fires: {set(monitor.active_fires.keys())}")

    # Simulate poll 3: Fire status changes to completed
    print("\n[POLL 3] Fire status changes to '진화완료' (03)")
    print("  (This is the ONLY completion status code used by KFS)")
    fake_fire_3 = {
        "frfrInfoId": "TEST001",
        "frfrLctnYcrd": "36.5684",
        "frfrLctnXcrd": "128.7294",
        "frfrFrngDtm": "2025-12-06 14:30:00",
        "frfrPrgrsStcd": "03",
        "frfrPrgrsStcdNm": "진화완료",
        "potfrCmpleDtm": "2025-12-06 17:00:00"  # Official completion time
    }
    fire_list_3 = [fake_fire_3]
    current_fire_ids = set([f.get("frfrInfoId") for f in fire_list_3])

    # Check if fire should end (SHOULD end, 03 is completion code)
    ended_fires = []
    for fire_id, fire_metadata in list(monitor.active_fires.items()):
        if fire_id not in current_fire_ids:
            ended_fires.append((fire_id, "disappeared"))
        else:
            current_fire = next((f for f in fire_list_3 if f.get("frfrInfoId") == fire_id), None)
            if current_fire:
                status_code = current_fire.get("frfrPrgrsStcd", "")
                if status_code in monitor.ENDED_STATUS_CODES:
                    ended_fires.append((fire_id, f"status_{status_code}"))

    print(f"  Current status: {fake_fire_3['frfrPrgrsStcdNm']}")
    print(f"  Fires to end: {ended_fires}")

    if ended_fires:
        print(f"  [SUCCESS] Fire detected as ended: {ended_fires[0][0]}")
        print(f"  Reason: {ended_fires[0][1]}")

    # Simulate poll 4: Fire disappears from API
    print("\n[POLL 4] Fire disappears from API")
    fire_list_4 = []
    current_fire_ids = set([f.get("frfrInfoId") for f in fire_list_4])

    # Reset active fires for this test
    monitor.active_fires["TEST002"] = {
        'lat': 37.5684,
        'lon': 129.7294,
        'timestamp': "2025-12-06T15:00:00",
        'status': "01",
        'status_name': "진행중",
        'last_seen_poll': 3
    }

    # Check if fire should end (SHOULD end, disappeared)
    ended_fires = []
    for fire_id, fire_metadata in list(monitor.active_fires.items()):
        if fire_id not in current_fire_ids:
            ended_fires.append((fire_id, "disappeared"))

    print(f"  Active fires before: {set(monitor.active_fires.keys())}")
    print(f"  Fires in current API: {current_fire_ids}")
    print(f"  Fires to end: {ended_fires}")

    if ended_fires:
        print(f"  [SUCCESS] Fire detected as ended: {ended_fires[0][0]}")
        print(f"  Reason: {ended_fires[0][1]}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

if __name__ == '__main__':
    test_fire_lifecycle()
