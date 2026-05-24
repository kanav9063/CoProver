#!/usr/bin/env bash
# prepare_all.sh — Download all datasets and create data files.
# Designed to run inside the SLIME docker container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

echo "=== Dataset Preparation ==="
echo "Script directory: ${SCRIPT_DIR}"
echo "Data directory:   ${DATA_DIR}"
echo ""

# Ensure data directory exists
mkdir -p "${DATA_DIR}"

# Install datasets library if not present
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing HuggingFace datasets library ..."
    pip install datasets
fi

# 1. Download MiniF2F test and Kimina-Prover-Promptset
echo ""
echo "=== Step 1: Downloading MiniF2F test & Kimina-Prover-Promptset ==="
python "${SCRIPT_DIR}/prepare_minif2f.py" --output-dir "${DATA_DIR}"

# 2. Convert trajectory data if available
TRAJECTORY_FILE="${DATA_DIR}/trajectories.jsonl"
if [ -f "${TRAJECTORY_FILE}" ]; then
    echo ""
    echo "=== Step 2: Converting trajectory data to value model SFT format ==="
    python "${SCRIPT_DIR}/prepare_value_data.py" \
        --input "${TRAJECTORY_FILE}" \
        --output "${DATA_DIR}/value_sft.jsonl" \
        --oversample-factor 3.0
else
    echo ""
    echo "=== Step 2: Skipping value data preparation ==="
    echo "No trajectory file found at ${TRAJECTORY_FILE}"
    echo "Run trajectory_collector.py first, then:"
    echo "  python prepare_value_data.py --input <trajectory.jsonl> --output ${DATA_DIR}/value_sft.jsonl --oversample-factor 3.0"
fi

echo ""
echo "=== Summary ==="
echo "Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}/"
echo ""
echo "Done."
