"""
Configuration for Fire Monitoring System
"""

# KFS API (Korea Forest Service)
KFS_REALTIME_URL = "https://fd.forest.go.kr/ffas/pubConn/selectPublicFireShowList.do"
KFS_POLL_INTERVAL = 120 # 2 minutes in seconds

# Flask Inference Server (running on same machine)
FLASK_SERVER_URL = "http://localhost:5000"
FLASK_PREDICT_ENDPOINT = f"{FLASK_SERVER_URL}/predict"
FLASK_HEALTH_ENDPOINT = f"{FLASK_SERVER_URL}/health"

# External Production Backend (TODO: Replace with actual URL)
EXTERNAL_BACKEND_URL = "https://api.example.com/wildfire/predictions"
EXTERNAL_BACKEND_TIMEOUT = 30  # seconds

# Monitoring Settings
LOG_FILE = "inference/fire_monitor/logs/monitor.log"
PROCESSED_FIRES_DB = "inference/fire_monitor/processed_fires.json"
