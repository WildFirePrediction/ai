#!/bin/bash

# Master script to run all data embedding scripts in sequence
# Updated to use Python scripts for easier debugging and modification

echo "=========================================="
echo "Wildfire Data Embedding Pipeline"
echo "=========================================="
echo ""

# Create output directory
mkdir -p ../embedded_data
mkdir -p logs

echo "Starting data embedding pipeline..."
echo ""

# 1. NASA VIIRS
echo "[1/7] Processing NASA VIIRS wildfire data (with pre-clustered episodes)..."
python3 01_nasa_viirs_embedding.py 2>&1 | tee logs/01_nasa_viirs.log

if [ $? -ne 0 ]; then
    echo "Error in NASA VIIRS processing. Check logs/01_nasa_viirs.log"
    exit 1
fi
echo "✓ NASA VIIRS processing completed"
echo ""

# 2. DEM & RSP
echo "[2/7] Processing DEM and RSP data..."
python3 02_dem_rsp_embedding.py 2>&1 | tee logs/02_dem_rsp.log

if [ $? -ne 0 ]; then
    echo "Error in DEM/RSP processing. Check logs/02_dem_rsp.log"
    exit 1
fi
echo "✓ DEM/RSP processing completed"
echo ""

# 3. Land Cover Map
echo "[3/7] Processing Land Cover Map..."
python3 03_lcm_embedding.py 2>&1 | tee logs/03_lcm.log

if [ $? -ne 0 ]; then
    echo "Error in LCM processing. Check logs/03_lcm.log"
    exit 1
fi
echo "✓ LCM processing completed"
echo ""

# 4. Forest Stand Map
echo "[4/7] Processing Forest Stand Map..."
python3 04_fsm_embedding.py 2>&1 | tee logs/04_fsm.log

if [ $? -ne 0 ]; then
    echo "Error in FSM processing. Check logs/04_fsm.log"
    exit 1
fi
echo "✓ FSM processing completed"
echo ""

# 5. NDVI
echo "[5/7] Processing NDVI data..."
python3 05_ndvi_embedding.py 2>&1 | tee logs/05_ndvi.log

if [ $? -ne 0 ]; then
    echo "Error in NDVI processing. Check logs/05_ndvi.log"
    exit 1
fi
echo "✓ NDVI processing completed"
echo ""

# 6. KMA Weather
echo "[6/7] Processing KMA weather data..."
python3 06_kma_weather_embedding.py 2>&1 | tee logs/06_kma_weather.log

if [ $? -ne 0 ]; then
    echo "Error in KMA weather processing. Check logs/06_kma_weather.log"
    exit 1
fi
echo "✓ KMA weather processing completed"
echo ""

# 7. Final State Composition
echo "[7/7] Creating final state composition..."
python3 07_final_state_composition.py 2>&1 | tee logs/07_final_composition.log

if [ $? -ne 0 ]; then
    echo "Error in final composition. Check logs/07_final_composition.log"
    exit 1
fi
echo "✓ Final composition completed"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output files located in: ../embedded_data/"
echo "Log files in: logs/"
echo ""
echo "Summary of outputs:"
ls -lh ../embedded_data/
echo ""
echo "Next steps:"
echo "1. Verify NASA VIIRS clustering results"
echo "2. Implement remaining data source scripts (LCM, FSM, NDVI)"
echo "3. Run final state composition"
echo "4. Ready for RL training!"

echo "✓ NASA VIIRS processing completed"
echo ""

# 2. DEM & RSP
echo "[2/7] Processing DEM and RSP data..."
papermill 02_dem_rsp_embedding.ipynb \
    notebook_outputs/02_dem_rsp_embedding_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in DEM/RSP processing. Exiting."
    exit 1
fi
echo "✓ DEM/RSP processing completed"
echo ""

# 3. Land Cover Map
echo "[3/7] Processing Land Cover Map..."
papermill 03_lcm_embedding.ipynb \
    notebook_outputs/03_lcm_embedding_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in LCM processing. Exiting."
    exit 1
fi
echo "✓ LCM processing completed"
echo ""

# 4. Forest Stand Map
echo "[4/7] Processing Forest Stand Map..."
papermill 04_fsm_embedding.ipynb \
    notebook_outputs/04_fsm_embedding_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in FSM processing. Exiting."
    exit 1
fi
echo "✓ FSM processing completed"
echo ""

# 5. NDVI
echo "[5/7] Processing NDVI data..."
papermill 05_ndvi_embedding.ipynb \
    notebook_outputs/05_ndvi_embedding_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in NDVI processing. Exiting."
    exit 1
fi
echo "✓ NDVI processing completed"
echo ""

# 6. KMA Weather
echo "[6/7] Processing KMA weather data..."
papermill 06_kma_weather_embedding.ipynb \
    notebook_outputs/06_kma_weather_embedding_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in KMA weather processing. Exiting."
    exit 1
fi
echo "✓ KMA weather processing completed"
echo ""

# 7. Final State Composition
echo "[7/7] Creating final state composition..."
papermill 07_final_state_composition.ipynb \
    notebook_outputs/07_final_state_composition_output.ipynb \
    --log-output

if [ $? -ne 0 ]; then
    echo "Error in final state composition. Exiting."
    exit 1
fi
echo "✓ Final state composition completed"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output files located in: embedded_data/"
echo "Executed notebooks in: notebook_outputs/"
echo ""
echo "Summary of outputs:"
ls -lh embedded_data/
echo ""
echo "Ready for RL training!"

