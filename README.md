# WildFirePrediction/ai

#### AI Module for Real-Time Wildfire Spread Prediction

This repository contains the AI components for a real-time wildfire spread prediction system, powered by geospatial data pipelines, reinforcement learning, and satellite-based fire detection feeds.


## Features

> - High resolution (300m) wildfire spread forecasting  
> 
> - RL based propagation model (A3C) trained on 10 years of data  
> 
> - Real time monitoring mode integrated with KFS(산림청) fire reports  
> 
> - Demo mode with synthetic ignition events
> 
> - Systemd service deployment for background inference and 24/7 monitoring


## Project Layout

```
WildfirePrediction
 ├── inference/
 │        ├──rl/                # RL model inference
 │        ├──sl/                # SL model inference 
 │        ├──demo_rl/           # Demo mode for RL model
 │        ├──demo_rl_multi/     # Demo mode for RL model (Multi-step)
 │        ├──demo_sl/           # Demo mode for SL model        
 │        ├──demo_sl_multi/     # Demo mode for SL model (Multi-step)
 │        └──fire_monitor/      # KFS API monitoring
 ├── rl_training/               # Reinforcement learning training
 │        ├──a3c_10ch/          # 10-channel A3C model
 │        ├──a3c_16ch/          # 16-channel A3C model (Production)
 │        └──...
 ├── sl_training/               # Supervised learning training
 │        ├──ag_unet_16ch/      # Spatial Attention U-Net model (Production)
 │        ├──unet_16ch/
 │        ├──unet_16ch_v2/
 │        ├──unet_16ch_v3/
 │        └──...
 ├── src/                       # Common utility scripts
 ├── deployment/                # systemd service files
 ├── embedding_src/             # Data embedding scripts
 ├── tilling_src/               # Environment tiling scripts
 │
 ├── README.md                  # This file
 ├── requirements.txt           # Python dependencies
 ├── download_data.sh           # Data download script (~1.6GB)
 ├── install_env.sh             # CUDA + NVIDIA driver install script
 ├── start_demo.sh              # Start demo mode script
 └── start_monitoring.sh        # Start production monitoring mode script

```

## Quick Start

### 1) Clone the Repository

- Renaming repo to `WildfirePrediction` is optional, but recommended for clarity

```bash
git clone https://github.com/WildFirePrediction/ai.git WildFirePrediction
cd WildFirePrediction
```

### 2) Download Required Data (~1.6GB)

- script to download **embedding data** to construct **environment tiles** for inference
- **google drive (wget)**

```bash
./download_data.sh
```

# Running the Wildfire Prediction

> **Tested Environment**
> - Ubuntu 24.04.3 LTS  
> - CUDA 13.0  
> - NVIDIA Driver 580.95.05  

---

## 0. (Recommended) Install CUDA + NVIDIA Driver

- Optional, but recommended to match tested environment

```bash
./install_env.sh
```


## 1. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 2. Run Inference

## 2-1. Demo Mode (Fake Fire Data)

```bash
./start_demo.sh
```

- Generates synthetic ignition every 120 seconds and runs full inference.
- Creates html visualization and JSON output for each inference.


```
WildfirePrediction
 └── inference/ 
         └──demo_rl/
               └──outputs/
                     ├──*.html
                     └──*.json
```


## 2-2. Production Mode (Real KFS API Monitoring)

```bash
./start_monitoring.sh
```

- Polls KFS API for new fire detections  
- Runs wildfire spread inference  
- Sends results to production backend 

```bash
# Configure backend URL in .env
EXTERNAL_BACKEND_URL=https://api.example.com/wildfire/predictions
```


# Background Deployment (systemd)

### 1. Install Services

```bash
sudo cp deployment/wildfire-api.service /etc/systemd/system/
sudo cp deployment/wildfire-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 2. Start Services

```bash
sudo systemctl start wildfire-api
sudo systemctl start wildfire-monitor
```

### 3. Stop Services

```bash
sudo systemctl stop wildfire-api
sudo systemctl stop wildfire-monitor
```


## Development Notes

- This repository contains only the AI inference engine.  
- Due to file size limits, training data is maintained [elsewhere](https://huggingface.co/datasets/chaseungjoon/wildfire-korea-episodes-300m).

