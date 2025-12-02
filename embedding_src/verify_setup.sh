#!/bin/bash
# Verify all paths and dependencies are correct before running pipeline

echo "🔍 Verifying Pipeline Setup..."
echo ""

ERRORS=0

# Check Python scripts
echo "[1/5] Checking Python scripts..."
for i in 01 02 03 04 05 06 07; do
    script=$(ls ${i}_*.py 2>/dev/null)
    if [ -n "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ Script ${i}_*.py missing!"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check data directories
echo ""
echo "[2/5] Checking data directories..."
REQUIRED_DIRS=(
    "../data/NASA/VIIRS"
    "../data/DigitalElevationModel"
    "../data/LandCoverMap"
    "../data/ForestStandMap"
    "../data/KMA"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "  ✓ $dir ($count files)"
    else
        echo "  ✗ $dir missing!"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check Python dependencies
echo ""
echo "[3/5] Checking Python dependencies..."
python3 << 'EOF'
import sys
required = ['numpy', 'pandas', 'rasterio', 'geopandas', 'pyproj', 'scipy', 'tqdm']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} missing!")
        missing.append(pkg)
sys.exit(len(missing))
EOF

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# Check critical data files
echo ""
echo "[4/5] Checking critical data files..."

# NASA clustered fires
if ls ../data/NASA/VIIRS/*/clustered_fire_archive_*.csv >/dev/null 2>&1; then
    count=$(ls ../data/NASA/VIIRS/*/clustered_fire_archive_*.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ NASA clustered fire archives ($count files)"
else
    echo "  ✗ No NASA clustered fire archives found!"
    ERRORS=$((ERRORS + 1))
fi

# DEM
if [ -f "../data/DigitalElevationModel/90m_GRS80.tif" ] || [ -f "../data/DigitalElevationModel/90m_GRS80.img" ]; then
    echo "  ✓ DEM file found"
else
    echo "  ✗ DEM file missing (90m_GRS80.tif or .img)!"
    ERRORS=$((ERRORS + 1))
fi

# LCM shapefiles
if find ../data/LandCoverMap -name "*.shp" 2>/dev/null | head -1 | grep -q .; then
    count=$(find ../data/LandCoverMap -name "*.shp" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Land Cover Map shapefiles ($count files)"
else
    echo "  ✗ No Land Cover Map shapefiles found!"
    ERRORS=$((ERRORS + 1))
fi

# FSM shapefiles
if find ../data/ForestStandMap -name "*.shp" 2>/dev/null | head -1 | grep -q .; then
    count=$(find ../data/ForestStandMap -name "*.shp" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Forest Stand Map shapefiles ($count files)"
else
    echo "  ✗ No Forest Stand Map shapefiles found!"
    ERRORS=$((ERRORS + 1))
fi

# KMA data
if find ../data/KMA -name "AWS_*.csv" 2>/dev/null | head -1 | grep -q .; then
    count=$(find ../data/KMA -name "AWS_*.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ KMA weather data ($count files)"
else
    echo "  ⚠ No KMA weather data found (optional)"
fi

# Check output directory
echo ""
echo "[5/5] Checking output directory..."
if [ ! -d "../embedded_data" ]; then
    mkdir -p ../embedded_data
    echo "  ✓ Created ../embedded_data/"
else
    echo "  ✓ ../embedded_data/ exists"
fi

if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "  ✓ Created logs/"
else
    echo "  ✓ logs/ exists"
fi

# Summary
echo ""
echo "=" * 60
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed! Pipeline is ready to run."
    echo ""
    echo "To start processing:"
    echo "  ./run_embedding_pipeline.sh"
    exit 0
else
    echo "❌ Found $ERRORS error(s). Please fix before running pipeline."
    echo ""
    echo "Common fixes:"
    echo "  - Install missing dependencies: pip install -r requirements.txt"
    echo "  - Verify data directory structure matches documentation"
    echo "  - Check data files are accessible"
    exit 1
fi

