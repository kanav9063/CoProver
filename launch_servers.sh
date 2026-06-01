#!/bin/bash
# launch_servers.sh — Start all servers needed for training and evaluation.
#
# Servers:
#   1. kimina-lean-server  (Docker, port 8000)  — Lean4 proof verification
#   2. SGLang generator    (port 30000)          — tactic/proof generation
#   3. SGLang value model  (port 30001)          — value estimation
#
# Usage:
#   bash launch_servers.sh                # start all servers
#   bash launch_servers.sh --kimina-only  # start only kimina-lean-server
#   bash launch_servers.sh --sglang-only  # start only SGLang servers
#
# Override defaults via environment:
#   KIMINA_PORT=8000  SGLANG_GEN_PORT=30000  SGLANG_VAL_PORT=30001
#   GENERATOR_MODEL=/workspace/models/DeepSeek-Prover-V2-7B
#   VALUE_MODEL=/workspace/models/Llama-3.2-1B

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE="${WORKSPACE:-/mnt/filesystem-m5/formal}"
TRAINING_DIR="${WORKSPACE}/training"
KIMINA_IMAGE="${KIMINA_IMAGE:-kimina-lean-server:latest}"
KIMINA_CONTAINER="${KIMINA_CONTAINER:-kimina-lean-server}"
KIMINA_PORT="${KIMINA_PORT:-8000}"
KIMINA_GPU="${KIMINA_GPU:-0}"

SGLANG_GEN_PORT="${SGLANG_GEN_PORT:-30000}"
SGLANG_VAL_PORT="${SGLANG_VAL_PORT:-30001}"
GENERATOR_MODEL="${GENERATOR_MODEL:-${WORKSPACE}/models/DeepSeek-Prover-V2-7B}"
VALUE_MODEL="${VALUE_MODEL:-${WORKSPACE}/models/Llama-3.2-1B}"
SGLANG_GEN_DP="${SGLANG_GEN_DP:-8}"
SGLANG_VAL_DP="${SGLANG_VAL_DP:-1}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
DOCKER_CMD=()
PYTHON_BIN=""

usage() {
    cat <<'EOF'
Usage:
  bash launch_servers.sh                # start all servers
  bash launch_servers.sh --kimina-only  # start only kimina-lean-server
  bash launch_servers.sh --sglang-only  # start only SGLang servers
EOF
}

# Parse flags
START_KIMINA=true
START_SGLANG=true

for arg in "$@"; do
    case "$arg" in
        --kimina-only) START_SGLANG=false ;;
        --sglang-only) START_KIMINA=false ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'time-unavailable')"
    echo "[${ts}] $*"
}

die() {
    log "FATAL: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

init_docker_bin() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        DOCKER_CMD=(docker)
        return 0
    fi

    if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
        DOCKER_CMD=(sudo docker)
        return 0
    fi

    die "Docker is unavailable. Install Docker or ensure this user can run 'docker info' (with or without passwordless sudo)."
}

init_python_bin() {
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
        return 0
    fi

    die "Python interpreter not found. Install python3 or ensure 'python' is on PATH."
}

init_environment() {
    require_cmd curl
    require_cmd grep
    require_cmd nohup
    init_docker_bin
    init_python_bin

    mkdir -p "${TRAINING_DIR}" || die "Could not create training directory: ${TRAINING_DIR}"
}

wait_for_health() {
    local name="$1"
    local url="$2"
    local alt_url="${3:-}"
    local timeout="${4:-$HEALTH_TIMEOUT}"

    log "Waiting for ${name} to be healthy (timeout: ${timeout}s) ..."
    local deadline=$((SECONDS + timeout))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            log "${name} is ready."
            return 0
        fi
        if [ -n "$alt_url" ] && curl -sf "${alt_url}" >/dev/null 2>&1; then
            log "${name} is ready."
            return 0
        fi
        sleep 5
    done

    log "WARNING: ${name} did not become healthy within ${timeout}s."
    return 1
}

# ---------------------------------------------------------------------------
# 1. kimina-lean-server (Docker)
# ---------------------------------------------------------------------------

start_kimina() {
    log "--- Starting kimina-lean-server ---"

    if "${DOCKER_CMD[@]}" ps --format '{{.Names}}' | grep -q "^${KIMINA_CONTAINER}$"; then
        log "kimina-lean-server is already running."
        wait_for_health "kimina-lean-server" \
            "http://localhost:${KIMINA_PORT}/health" \
            "http://localhost:${KIMINA_PORT}/api/check" \
            30
        return $?
    fi

    # Remove any stopped container with the same name
    if "${DOCKER_CMD[@]}" ps -a --format '{{.Names}}' | grep -q "^${KIMINA_CONTAINER}$"; then
        log "Removing stopped kimina-lean-server container ..."
        "${DOCKER_CMD[@]}" rm "${KIMINA_CONTAINER}" >/dev/null 2>&1
    fi

    log "Starting kimina-lean-server container (image: ${KIMINA_IMAGE}, GPU: ${KIMINA_GPU}) ..."
    "${DOCKER_CMD[@]}" run -d \
        --name "${KIMINA_CONTAINER}" \
        --gpus "device=${KIMINA_GPU}" \
        -p "${KIMINA_PORT}:8000" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --restart unless-stopped \
        "${KIMINA_IMAGE}" || {
            log "ERROR: Could not start kimina-lean-server."
            return 1
        }

    wait_for_health "kimina-lean-server" \
        "http://localhost:${KIMINA_PORT}/health" \
        "http://localhost:${KIMINA_PORT}/api/check" \
        "${HEALTH_TIMEOUT}"
}

# ---------------------------------------------------------------------------
# 2. SGLang Generator Server
# ---------------------------------------------------------------------------

start_sglang_generator() {
    log "--- Starting SGLang generator server ---"

    if curl -sf "http://localhost:${SGLANG_GEN_PORT}/health" >/dev/null 2>&1 || \
       curl -sf "http://localhost:${SGLANG_GEN_PORT}/v1/models" >/dev/null 2>&1; then
        log "SGLang generator is already running on port ${SGLANG_GEN_PORT}."
        return 0
    fi

    if [ ! -d "${GENERATOR_MODEL}" ]; then
        die "Generator model not found: ${GENERATOR_MODEL}"
    fi

    log "Launching SGLang generator (model: ${GENERATOR_MODEL}, port: ${SGLANG_GEN_PORT}, dp: ${SGLANG_GEN_DP}) ..."
    nohup "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "${GENERATOR_MODEL}" \
        --port "${SGLANG_GEN_PORT}" \
        --dp "${SGLANG_GEN_DP}" \
        > "${TRAINING_DIR}/sglang_generator.log" 2>&1 &

    local pid=$!
    log "SGLang generator PID: ${pid}"
    echo "${pid}" > "${TRAINING_DIR}/.sglang_generator.pid"

    wait_for_health "SGLang generator" \
        "http://localhost:${SGLANG_GEN_PORT}/health" \
        "http://localhost:${SGLANG_GEN_PORT}/v1/models" \
        "${HEALTH_TIMEOUT}"
}

# ---------------------------------------------------------------------------
# 3. SGLang Value Model Server
# ---------------------------------------------------------------------------

start_sglang_value() {
    log "--- Starting SGLang value model server ---"

    if curl -sf "http://localhost:${SGLANG_VAL_PORT}/health" >/dev/null 2>&1 || \
       curl -sf "http://localhost:${SGLANG_VAL_PORT}/v1/models" >/dev/null 2>&1; then
        log "SGLang value model is already running on port ${SGLANG_VAL_PORT}."
        return 0
    fi

    if [ ! -d "${VALUE_MODEL}" ]; then
        die "Value model not found: ${VALUE_MODEL}"
    fi

    log "Launching SGLang value model (model: ${VALUE_MODEL}, port: ${SGLANG_VAL_PORT}, dp: ${SGLANG_VAL_DP}) ..."
    nohup "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "${VALUE_MODEL}" \
        --port "${SGLANG_VAL_PORT}" \
        --dp "${SGLANG_VAL_DP}" \
        > "${TRAINING_DIR}/sglang_value.log" 2>&1 &

    local pid=$!
    log "SGLang value model PID: ${pid}"
    echo "${pid}" > "${TRAINING_DIR}/.sglang_value.pid"

    wait_for_health "SGLang value model" \
        "http://localhost:${SGLANG_VAL_PORT}/health" \
        "http://localhost:${SGLANG_VAL_PORT}/v1/models" \
        "${HEALTH_TIMEOUT}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ "$START_KIMINA" = false ] && [ "$START_SGLANG" = false ]; then
    die "choose either --kimina-only or --sglang-only, not both"
fi

log "=== Server Launcher ==="
init_environment

FAILED=0

if [ "$START_KIMINA" = true ]; then
    start_kimina || FAILED=$((FAILED + 1))
fi

if [ "$START_SGLANG" = true ]; then
    start_sglang_generator || FAILED=$((FAILED + 1))
    start_sglang_value     || FAILED=$((FAILED + 1))
fi

echo ""
log "=== Summary ==="
if [ "$START_KIMINA" = true ]; then
    if curl -sf "http://localhost:${KIMINA_PORT}/health" >/dev/null 2>&1 || \
       curl -sf "http://localhost:${KIMINA_PORT}/api/check" >/dev/null 2>&1; then
        log "  kimina-lean-server : UP (port ${KIMINA_PORT})"
    else
        log "  kimina-lean-server : DOWN (port ${KIMINA_PORT})"
    fi
fi
if [ "$START_SGLANG" = true ]; then
    if curl -sf "http://localhost:${SGLANG_GEN_PORT}/v1/models" >/dev/null 2>&1; then
        log "  SGLang generator   : UP (port ${SGLANG_GEN_PORT})"
    else
        log "  SGLang generator   : DOWN (port ${SGLANG_GEN_PORT})"
    fi
    if curl -sf "http://localhost:${SGLANG_VAL_PORT}/v1/models" >/dev/null 2>&1; then
        log "  SGLang value model : UP (port ${SGLANG_VAL_PORT})"
    else
        log "  SGLang value model : DOWN (port ${SGLANG_VAL_PORT})"
    fi
fi

if [ $FAILED -gt 0 ]; then
    log "${FAILED} server(s) failed to start. Check logs above."
    exit 1
fi

log "All servers are up."
