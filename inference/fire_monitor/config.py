import os

from dotenv import load_dotenv
load_dotenv()

"""
Configuration for Fire Monitoring System
"""

# KFS API (Korea Forest Service)
KFS_REALTIME_URL = "https://fd.forest.go.kr/ffas/pubConn/selectPublicFireShowList.do"
KFS_POLL_INTERVAL = 300

# Flask Inference Server (running on same machine)
FLASK_SERVER_URL = "http://localhost:5000"
FLASK_PREDICT_ENDPOINT = f"{FLASK_SERVER_URL}/predict"
FLASK_HEALTH_ENDPOINT = f"{FLASK_SERVER_URL}/health"

# External Production Backend (supports multiple URLs)
# Format: Single URL or comma-separated URLs
# Example: "https://backend1.com,https://backend2.com"
_backend_url_str = os.getenv("EXTERNAL_BACKEND_URL", "")
if _backend_url_str:
    # Split by comma and strip whitespace
    EXTERNAL_BACKEND_URLS = [url.strip()
                             for url in _backend_url_str.split(",") if url.strip()]
else:
    EXTERNAL_BACKEND_URLS = []

# Legacy: Keep single URL for backward compatibility
EXTERNAL_BACKEND_URL = EXTERNAL_BACKEND_URLS[0] if EXTERNAL_BACKEND_URLS else None

EXTERNAL_BACKEND_TIMEOUT = 30  # seconds

# Monitoring Settings
LOG_FILE = "inference/fire_monitor/logs/monitor.log"
PROCESSED_FIRES_DB = "inference/fire_monitor/processed_fires.json"
