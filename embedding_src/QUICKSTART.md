# Data Embedding Quickstart

## Prerequisites

- **OS:** Ubuntu 24.04 LTS
- **Python:** 3.10+
- **Hardware:** Ryzen9 9950X, 64GB DDR5 RAM
- **GPU:** RTX5070 (CUDA 12.0+)

## Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y gdal-bin libgdal-dev libspatialindex-dev python3-dev python3-pip
```

## Step 2: Setup Python Environment

```bash
cd embedding_src

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
cd .. && pip install -r requirements.txt
cd embedding_src
```

## Step 3: Verify Data Structure

```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

## Step 4: Run the Pipeline

```bash
chmod +x run_embedding_pipeline.sh

# Run complete pipeline (takes 1-3 hours)
./run_embedding_pipeline.sh
```

**Or run scripts individually:**

```bash
python3 01_nasa_viirs_embedding.py
python3 02_dem_rsp_embedding.py
python3 03_lcm_embedding.py
python3 04_fsm_embedding.py
python3 05_ndvi_embedding.py
python3 06_kma_weather_embedding.py
python3 07_final_state_composition.py
```

### Step 5: Verify Outputs

```bash
# Check output files
ls -lh ../embedded_data/

# Should see 15 files including:
# - state_vectors.npz (main output)
# - episode_index.parquet (fire episodes)
# - *.tif (spatial grids)
# - *.json (metadata)
```

---

## Main Output: `state_vectors.npz`

```python
import numpy as np

# Load final state vectors
data = np.load('../embedded_data/state_vectors.npz')

# Available arrays:
print(data['continuous_features'].shape)  # (13, H, W) continuous features
print(data['feature_names'])              # List of feature names
print(data['lcm_classes'].shape)          # (H, W) land cover classes
print(data['fsm_classes'].shape)          # (H, W) forest stand types
```

### Features Included

**Continuous (13 features):**
- x_norm, y_norm - Spatial coordinates
- te_norm, i_norm, tm_norm - Fire characteristics
- dem_norm, rsp_norm - Topography
- ndvi_norm - Vegetation
- w_norm, d_x_norm, d_y_norm - Wind
- rh_norm, r_norm - Weather

**Categorical (2 features):**
- lcm - Land cover class IDs
- fsm - Forest stand type IDs

### Fire Episodes

```python
import pandas as pd

# Load fire episode index
episodes = pd.read_parquet('../embedded_data/episode_index.parquet')

print(f"Total episodes: {len(episodes)}")
print(f"Avg duration: {episodes['duration_hours'].mean():.1f} hours")
print(f"Avg detections: {episodes['num_detections'].mean():.1f}")
```

---

## Using Results with PyTorch

```python
import torch
import numpy as np

# Load embedded data
data = np.load('../embedded_data/state_vectors.npz')

# Create PyTorch tensors
continuous = torch.from_numpy(data['continuous_features']).float()
lcm = torch.from_numpy(data['lcm_classes']).long()
fsm = torch.from_numpy(data['fsm_classes']).long()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
continuous = continuous.to(device)

print(f"State shape: {continuous.shape}")
print(f"Device: {device}")
# Ready for A3C model!
```

---

## Expected Timeline

| Stage | Time | Output |
|-------|------|--------|
| 01 - NASA VIIRS | 10-20 min | Fire detections |
| 02 - DEM/RSP | 5-10 min | Topography grid |
| 03 - LCM | 20-40 min | Land cover |
| 04 - FSM | 30-60 min | Forest types |
| 05 - NDVI | 5-10 min | Vegetation |
| 06 - KMA | 10-15 min | Weather |
| 07 - Final | 5-10 min | State vectors |
| **Total** | **1.5-3 hours** | **~1.5 GB** |

---

## Success Checklist

After completion, verify:

```bash
# All output files present
ls ../embedded_data/ | wc -l

# Main output loads correctly
python3 -c "import numpy as np; d=np.load('../embedded_data/state_vectors.npz'); print('OK')"

# Fire episodes extracted
python3 -c "import pandas as pd; e=pd.read_parquet('../embedded_data/episode_index.parquet'); print(f'{len(e)} episodes')"

# Grid alignment verified
python3 << 'EOF'
import rasterio
files = ['dem_rsp', 'lcm', 'fsm', 'ndvi', 'kma_weather']
shapes = []
for f in files:
    try:
        with rasterio.open(f'../embedded_data/{f}_embedded.tif') as src:
            shapes.append(src.shape)
            print(f"{f}: {src.shape}")
    except: pass
print(f"✓ Aligned" if len(set(shapes)) == 1 else "✗ Mismatch")
EOF
```

---

## Next Steps

1. **Verify outputs** - Check all files created successfully
2. **Review episodes** - Examine fire episode statistics
3. **Build RL model** - Use state vectors for A3C training
4. **Iterate** - Adjust parameters and reprocess as needed
