#!/bin/bash
# =============================================================================
# Co-Training Loop -- runs INSIDE the SLIME Docker container
#
# This is the top-level entrypoint for the full co-training pipeline:
#   1. Pre-flight checks (kimina-lean-server, dependencies, GPUs)
#   2. Environment setup (WANDB, PYTHONPATH, elan/PATH)
#   3. Launches co_train.py with /workspace/-mapped paths
#   4. Traps SIGTERM/SIGINT to clean up SGLang, Ray, and child processes
#
# Usage (inside the container):
#   bash /workspace/training/co_train.sh
#   bash /workspace/training/co_train.sh --num-rounds 10 --start-round 3
#   NUM_ROUNDS=3 bash /workspace/training/co_train.sh
#
# Expected to be launched via:
#   docker run ... slimerl/slime:latest bash /workspace/training/co_train.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_PREFIX="[co_train.sh]"
readonly KIMINA_HEALTH_URL="${KIMINA_HEALTH_URL:-http://172.17.0.1:8000/health}"
readonly KIMINA_HEALTH_TIMEOUT="${KIMINA_HEALTH_TIMEOUT:-60}"
readonly KIMINA_URL="${KIMINA_URL:-http://172.17.0.1:8000}"

# Container paths
readonly GENERATOR_MODEL="/workspace/models/DeepSeek-Prover-V2-7B"
readonly VALUE_MODEL="/workspace/models/Llama-3.2-1B"
readonly TRAINING_DIR="/workspace/training"
readonly REPROVER_DATA="/workspace/ReProver/data"
readonly SLIME_ROOT="/root/slime"
readonly MEGATRON_ROOT="/root/Megatron-LM"
readonly ELAN_BIN="/root/.elan/bin"

# Tuneable defaults (override via environment)
NUM_ROUNDS="${NUM_ROUNDS:-5}"

# Extra CLI arguments forwarded to co_train.py
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} $*"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} WARNING: $*" >&2; }
die()  { echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Cleanup & signal handling
# ---------------------------------------------------------------------------

CHILD_PID=""

cleanup() {
    local exit_code=$?
    log "Cleanup triggered (exit_code=${exit_code})"

    # Kill the main python process if still running
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        log "Sending SIGTERM to co_train.py (pid ${CHILD_PID})"
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        # Give it a few seconds to shut down gracefully
        local wait_count=0
        while kill -0 "${CHILD_PID}" 2>/dev/null && (( wait_count < 15 )); do
            sleep 1
            (( wait_count++ ))
        done
        if kill -0 "${CHILD_PID}" 2>/dev/null; then
            log "co_train.py did not exit gracefully; sending SIGKILL"
            kill -9 "${CHILD_PID}" 2>/dev/null || true
        fi
    fi

    # Kill any lingering SGLang servers
    log "Cleaning up SGLang processes"
    pkill -9 -f "sglang" 2>/dev/null || true

    # Stop Ray
    log "Stopping Ray"
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray" 2>/dev/null || true

    # Kill any orphaned python GPU processes that might hold VRAM
    pkill -9 -f "python.*launch_server" 2>/dev/null || true

    log "Cleanup complete"
    exit "${exit_code}"
}

trap cleanup EXIT
trap 'log "Received SIGTERM"; exit 143' SIGTERM
trap 'log "Received SIGINT";  exit 130' SIGINT

# ---------------------------------------------------------------------------
# Pre-flight: kimina-lean-server health check
# ---------------------------------------------------------------------------

check_kimina() {
    log "Checking kimina-lean-server at ${KIMINA_HEALTH_URL}"

    local deadline=$(( SECONDS + KIMINA_HEALTH_TIMEOUT ))
    local attempt=0

    while (( SECONDS < deadline )); do
        (( attempt++ ))
        if curl --silent --fail --max-time 5 "${KIMINA_HEALTH_URL}" >/dev/null 2>&1; then
            log "kimina-lean-server is reachable (attempt ${attempt})"
            return 0
        fi
        if (( attempt == 1 )); then
            log "Waiting up to ${KIMINA_HEALTH_TIMEOUT}s for kimina-lean-server..."
        fi
        sleep 3
    done

    warn "kimina-lean-server did not respond within ${KIMINA_HEALTH_TIMEOUT}s"
    warn "Evaluation phases that depend on Lean compilation will be skipped."
    warn "To fix: ensure the kimina-lean-server container is running on the host."
    return 1
}

# ---------------------------------------------------------------------------
# Pre-flight: dependency checks inside container
# ---------------------------------------------------------------------------

check_dependencies() {
    log "Checking container dependencies..."
    local missing=()

    # Python packages
    for pkg in torch sglang wandb ray; do
        if ! python3 -c "import ${pkg}" 2>/dev/null; then
            missing+=("python:${pkg}")
        fi
    done

    # Filesystem paths
    for dir in "${SLIME_ROOT}" "${MEGATRON_ROOT}"; do
        if [[ ! -d "${dir}" ]]; then
            missing+=("dir:${dir}")
        fi
    done

    # Key binaries
    for bin in python3 ray curl; do
        if ! command -v "${bin}" >/dev/null 2>&1; then
            missing+=("bin:${bin}")
        fi
    done

    # GPU check
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        missing+=("bin:nvidia-smi")
    else
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
        log "GPUs detected: ${gpu_count}"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
        if (( gpu_count == 0 )); then
            die "No GPUs detected -- co-training requires GPUs"
        fi
    fi

    if (( ${#missing[@]} > 0 )); then
        die "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies present"
}

# ---------------------------------------------------------------------------
# Pre-flight: workspace paths
# ---------------------------------------------------------------------------

check_workspace() {
    log "Checking workspace paths..."

    if [[ ! -d "${TRAINING_DIR}" ]]; then
        die "Training directory not found: ${TRAINING_DIR}"
    fi

    if [[ ! -f "${TRAINING_DIR}/co_train.py" ]]; then
        die "co_train.py not found in ${TRAINING_DIR}"
    fi

    # Models -- warn but do not die (checkpoints from previous rounds may suffice)
    for model_dir in "${GENERATOR_MODEL}" "${VALUE_MODEL}"; do
        if [[ ! -d "${model_dir}" ]]; then
            warn "Model directory not found: ${model_dir}"
            warn "Training will fail unless a checkpoint from a previous round exists."
        fi
    done

    # Training scripts
    for script in train_step_grpo.sh train_value_slime.sh trajectory_collector.py evaluate.py; do
        if [[ ! -f "${TRAINING_DIR}/${script}" ]]; then
            warn "Expected script not found: ${TRAINING_DIR}/${script}"
        fi
    done

    # Data directories
    if [[ -d "${REPROVER_DATA}" ]]; then
        log "ReProver data: ${REPROVER_DATA}"
    else
        warn "ReProver data directory not found: ${REPROVER_DATA}"
    fi

    log "Workspace checks complete"
}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

setup_environment() {
    log "Configuring environment..."

    # W&B
    export WANDB_API_KEY="${WANDB_API_KEY:-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}"
    log "WANDB_API_KEY is set (length=${#WANDB_API_KEY})"

    # PYTHONPATH: SLIME, Megatron-LM, and workspace
    export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}:/workspace:${PYTHONPATH:-}"
    log "PYTHONPATH=${PYTHONPATH}"

    # PATH: ensure elan (Lean toolchain manager) is available
    if [[ -d "${ELAN_BIN}" ]]; then
        export PATH="${ELAN_BIN}:${PATH}"
        log "elan added to PATH ($(ls "${ELAN_BIN}" 2>/dev/null | head -5 | tr '\n' ' '))"
    else
        warn "elan not found at ${ELAN_BIN} -- Lean compilation may fail"
    fi

    # CUDA
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

    # Disable caching that can cause stale reads
    export DISABLE_REMOTE_CACHE="${DISABLE_REMOTE_CACHE:-1}"

    # Unbuffered output for real-time log streaming
    export PYTHONUNBUFFERED=1

    log "Environment configured"
}

# ---------------------------------------------------------------------------
# Build co_train.py argument list
# ---------------------------------------------------------------------------

build_args() {
    CO_TRAIN_ARGS=(
        --num-rounds "${NUM_ROUNDS}"
        --base-dir "${TRAINING_DIR}"
        --generator-model-hf "${GENERATOR_MODEL}"
        --value-model-hf "${VALUE_MODEL}"
        --theorems-path "${REPROVER_DATA}/leandojo_benchmark_4/random/train.json"
        --eval-dataset "${TRAINING_DIR}/data/minif2f.json"
        --grpo-script "${TRAINING_DIR}/train_step_grpo.sh"
        --value-script "${TRAINING_DIR}/train_value_slime.sh"
        --kimina-url "${KIMINA_URL}"
    )

    # Append any user-provided extra CLI arguments
    if (( ${#EXTRA_ARGS[@]} > 0 )); then
        CO_TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "========================================================"
    log "  Co-Training Loop -- SLIME Container Entrypoint"
    log "========================================================"
    log "Hostname: $(hostname)"
    log "Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    log "Script:   ${BASH_SOURCE[0]}"
    log ""

    # --- Pre-flight checks ---
    local kimina_ok=true
    check_kimina || kimina_ok=false
    check_dependencies
    check_workspace

    # --- Environment ---
    setup_environment

    # --- Arguments ---
    build_args

    # If kimina is not reachable, tell co_train.py to use a bogus URL so it
    # can detect the failure on its own instead of hanging.
    if [[ "${kimina_ok}" != "true" ]]; then
        warn "Passing --kimina-url with unreachable host; eval phases will detect and skip."
    fi

    # --- Launch ---
    log "========================================================"
    log "  Launching co_train.py"
    log "  Rounds: ${NUM_ROUNDS}"
    log "  Args:   ${CO_TRAIN_ARGS[*]}"
    log "========================================================"

    python3 "${TRAINING_DIR}/co_train.py" "${CO_TRAIN_ARGS[@]}" &
    CHILD_PID=$!

    log "co_train.py started (pid ${CHILD_PID})"

    # Wait for the child and propagate its exit code.
    # Using 'wait' allows the trap handlers to fire on signals.
    wait "${CHILD_PID}"
    exit_code=$?
    CHILD_PID=""

    if (( exit_code == 0 )); then
        log "co_train.py completed successfully"
    else
        log "co_train.py exited with code ${exit_code}"
    fi

    exit "${exit_code}"
}

main
