#!/usr/bin/env bash
set -euo pipefail

# Tilling pipeline runner for the whole dataset (per-episode regions)
# Steps:
#   01 - Episode region extraction (extent + 5-cell padding)
#   02 - Temporal segmentation (6h)
#   03 - Environment assembly (9-direction actions)
#   04 - Dataset split (train/val/test)
#   05 - Validation (light checks)
#
# Usage:
#   ./run_tilling.sh                        # run full dataset
#   ./run_tilling.sh --max-episodes 5       # quick smoke test on first 5 episodes
#   ./run_tilling.sh --clean                # clean previous tiling outputs then run full
#   ./run_tilling.sh --clean --max-episodes 5

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${THIS_DIR}/.." && pwd)"
LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAX_EPISODES_ARG=""
CLEAN="0"
# Simple arg parsing loop
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-episodes)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --max-episodes requires a value" >&2; exit 1
      fi
      MAX_EPISODES_ARG="--max-episodes $1"; shift ;;
    --clean)
      CLEAN="1"; shift ;;
    --help|-h)
      echo "Usage: $0 [--clean] [--max-episodes N]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

banner() {
  echo "================================================================================"
  echo "TILLING PIPELINE RUNNER"
  echo "Repo: ${REPO_DIR}"
  echo "Python: $(python3 -V 2>&1 | tr -d '\n')"
  echo "Date: $(date -Iseconds)"
  echo "================================================================================"
}

check_inputs() {
  echo "[CHECK] Verifying embedded inputs..."
  local ED="${REPO_DIR}/embedded_data"
  local required=(
    "${ED}/state_vectors.npz"
    "${ED}/grid_metadata.json"
    "${ED}/nasa_viirs_with_weather_reclustered.parquet"
    "${ED}/episode_index_reclustered.parquet"
  )
  local missing=0
  for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "  ✗ Missing: $f"
      missing=1
    else
      echo "  ✓ Found:   $f"
    fi
  done
  if [[ $missing -ne 0 ]]; then
    echo "ERROR: One or more required embedded inputs are missing. Aborting." >&2
    exit 1
  fi
}

pre_clean() {
  if [[ "$CLEAN" == "1" ]]; then
    echo "[CLEAN] Removing previous tiling outputs (regions, sequences, environments, legacy tiles/manifests)..."
    TD="${REPO_DIR}/tilling_data"
    rm -rf "${TD}/regions" "${TD}/sequences" "${TD}/environments" \
           "${TD}/episode_regions.parquet" "${TD}/episode_region_summary.json" \
           "${TD}/temporal_segments_summary.json" "${TD}/environment_assembly_summary.json" \
           "${TD}/tiles" "${TD}/tile_index.parquet" "${TD}/tile_index_all.parquet" \
           "${TD}/spatial_tiling_summary.json" "${TD}/tile_validation_report.json"
    mkdir -p "${TD}/regions" "${TD}/sequences" "${TD}/environments"
    echo "[CLEAN] Done. Starting fresh."
  fi
}

run_step() {
  local step="$1"; shift
  local script_path="${THIS_DIR}/${step}.py"
  local log_path="${LOG_DIR}/${step}.log"
  echo "\n>>> Running ${step}.py"
  echo "    Log: ${log_path}"
  time python3 "${script_path}" "$@" 2>&1 | tee "${log_path}"
}

main() {
  banner
  check_inputs
  pre_clean

  # Step 01: Episode region extraction
  run_step 01_spatial_tiling ${MAX_EPISODES_ARG}

  # Step 02: Temporal segmentation
  run_step 02_temporal_segmentation

  # Step 03: Environment assembly
  run_step 03_environment_assembly

  # Step 04: Dataset split
  run_step 04_dataset_split

  # Step 05: Validation
  run_step 05_environment_validation

  echo "\nAll steps completed. Outputs in: ${REPO_DIR}/tilling_data"
  echo "Logs in: ${LOG_DIR}"
}

main "$@"
