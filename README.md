# WildFirePrediction-AI

중앙대학교 25-2 캡스톤디자인 프로젝트로 진행되는 **산불 확산 예측 서비스 개발**의 AI Repo

## Download Project 

```bash
git clone https://github.com/WildFirePrediction/ai.git WildFirePrediction
cd WildFirePrediction
./download_data.sh
```

## Run Wildfire Prediction (backend server URL needed)

> Tested environment
>
> - Ubuntu 24.04.3 LTS
>
> - CUDA 13.0
>
> - NVIDIA Driver 580.95.05

### 0. (Recommended) Install tested CUDA and NVIDIA Driver

```bash
./install_env.sh
```

### 1. Create virtual environment & Install python packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2-1. Run demo with fake fire data

```bash
./start_demo.sh
```

- Generates fake fire every 120 seconds and feeds into inference pipeline
- Output saved at `~/inference/demo_rl/outputs` as `*.json / *.html` 

### 2-2. (WIP) Run production

```bash
./start_monitoring.sh
```

- Monitors KFS API for fire data
- Conducts inference and send result to production backend

### +) Run production in background

1) Setup

```bash
# Copy service files
sudo cp deployment/wildfire-api.service /etc/systemd/system/
sudo cp deployment/wildfire-monitor.service /etc/systemd/system/
```

```bash
# Reload systemd
sudo systemctl daemon-reload
```

2) Start services

```bash
sudo systemctl start wildfire-api
sudo systemctl start wildfire-monitor
```

3) Stop services

```bash
sudo systemctl stop wildfire-api
sudo systemctl stop wildfire-monitor
```


