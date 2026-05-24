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

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE="${WORKSPACE:-/mnt/filesystem-m5/formal}"
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

# Parse flags
START_KIMINA=true
START_SGLANG=true

for arg in "$@"; do
    case "$arg" in
        --kimina-only) START_SGLANG=false ;;
        --sglang-only) START_KIMINA=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "FATAL: $*" >&2
    exit 1
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

    if docker ps --format '{{.Names}}' | grep -q "^${KIMINA_CONTAINER}$"; then
        log "kimina-lean-server is already running."
        wait_for_health "kimina-lean-server" \
            "http://localhost:${KIMINA_PORT}/health" \
            "http://localhost:${KIMINA_PORT}/api/check" \
            30
        return $?
    fi

    # Remove any stopped container with the same name
    if docker ps -a --format '{{.Names}}' | grep -q "^${KIMINA_CONTAINER}$"; then
        log "Removing stopped kimina-lean-server container ..."
        docker rm "${KIMINA_CONTAINER}" >/dev/null 2>&1
    fi

    log "Starting kimina-lean-server container (image: ${KIMINA_IMAGE}, GPU: ${KIMINA_GPU}) ..."
    sudo docker run -d \
        --name "${KIMINA_CONTAINER}" \
        --gpus "\"device=${KIMINA_GPU}\"" \
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
    nohup python -m sglang.launch_server \
        --model-path "${GENERATOR_MODEL}" \
        --port "${SGLANG_GEN_PORT}" \
        --dp "${SGLANG_GEN_DP}" \
        > "${WORKSPACE}/training/sglang_generator.log" 2>&1 &

    local pid=$!
    log "SGLang generator PID: ${pid}"
    echo "${pid}" > "${WORKSPACE}/training/.sglang_generator.pid"

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
    nohup python -m sglang.launch_server \
        --model-path "${VALUE_MODEL}" \
        --port "${SGLANG_VAL_PORT}" \
        --dp "${SGLANG_VAL_DP}" \
        > "${WORKSPACE}/training/sglang_value.log" 2>&1 &

    local pid=$!
    log "SGLang value model PID: ${pid}"
    echo "${pid}" > "${WORKSPACE}/training/.sglang_value.pid"

    wait_for_health "SGLang value model" \
        "http://localhost:${SGLANG_VAL_PORT}/health" \
        "http://localhost:${SGLANG_VAL_PORT}/v1/models" \
        "${HEALTH_TIMEOUT}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "=== Server Launcher ==="

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
