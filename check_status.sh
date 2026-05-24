#!/bin/bash
# check_status.sh — Quick status check for the training pipeline.
#
# Shows GPU usage, server health, latest training metrics, and disk usage.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/mnt/filesystem-m5/formal}"
TRAINING_DIR="${WORKSPACE}/training"
KIMINA_PORT="${KIMINA_PORT:-8000}"
SGLANG_GEN_PORT="${SGLANG_GEN_PORT:-30000}"
SGLANG_VAL_PORT="${SGLANG_VAL_PORT:-30001}"

BOLD='\033[1m'
CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
RESET='\033[0m'

section() {
    echo ""
    echo -e "${BOLD}${CYAN}=== $1 ===${RESET}"
}

ok()   { echo -e "  ${GREEN}[OK]${RESET} $1"; }
fail() { echo -e "  ${RED}[FAIL]${RESET} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${RESET} $1"; }

# ---------------------------------------------------------------------------
# GPU Usage
# ---------------------------------------------------------------------------

section "GPU Usage"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=', ' read -r idx name util mem_used mem_total temp; do
        pct=$((mem_used * 100 / mem_total))
        echo -e "  GPU ${idx}: ${name}  |  util: ${util}%  |  mem: ${mem_used}/${mem_total} MiB (${pct}%)  |  temp: ${temp}C"
    done
else
    warn "nvidia-smi not found"
fi

# ---------------------------------------------------------------------------
# Server Health
# ---------------------------------------------------------------------------

section "Server Health"

# kimina-lean-server
if curl -sf "http://localhost:${KIMINA_PORT}/health" >/dev/null 2>&1 || \
   curl -sf "http://localhost:${KIMINA_PORT}/api/check" >/dev/null 2>&1; then
    ok "kimina-lean-server (port ${KIMINA_PORT})"
else
    # Check if Docker container is at least running
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q kimina-lean-server; then
        warn "kimina-lean-server container running but health check failed (port ${KIMINA_PORT})"
    else
        fail "kimina-lean-server not running (port ${KIMINA_PORT})"
    fi
fi

# SGLang generator
if curl -sf "http://localhost:${SGLANG_GEN_PORT}/health" >/dev/null 2>&1 || \
   curl -sf "http://localhost:${SGLANG_GEN_PORT}/v1/models" >/dev/null 2>&1; then
    ok "SGLang generator (port ${SGLANG_GEN_PORT})"
else
    fail "SGLang generator not running (port ${SGLANG_GEN_PORT})"
fi

# SGLang value model
if curl -sf "http://localhost:${SGLANG_VAL_PORT}/health" >/dev/null 2>&1 || \
   curl -sf "http://localhost:${SGLANG_VAL_PORT}/v1/models" >/dev/null 2>&1; then
    ok "SGLang value model (port ${SGLANG_VAL_PORT})"
else
    fail "SGLang value model not running (port ${SGLANG_VAL_PORT})"
fi

# Ray cluster
if ray status &>/dev/null; then
    ok "Ray cluster"
else
    fail "Ray cluster not running"
fi

# ---------------------------------------------------------------------------
# Latest Training Step
# ---------------------------------------------------------------------------

section "Latest Training Step (from logs)"

# Look for the most recent ray job log or any .log file
find_latest_log() {
    # Check ray job logs first
    local ray_log
    ray_log=$(find /tmp/ray -name "*.log" -newer "${TRAINING_DIR}/Makefile" -type f 2>/dev/null | \
              xargs ls -t 2>/dev/null | head -1)
    if [ -n "$ray_log" ]; then
        echo "$ray_log"
        return
    fi
    # Fall back to any log in training dir
    ls -t "${TRAINING_DIR}"/*.log 2>/dev/null | head -1
}

LATEST_LOG=$(find_latest_log)
if [ -n "$LATEST_LOG" ]; then
    echo "  Log: ${LATEST_LOG}"
    # Extract latest step/iteration info
    STEP_LINE=$(grep -oP '(step|iter|iteration)[=: ]+\d+' "$LATEST_LOG" 2>/dev/null | tail -1)
    if [ -n "$STEP_LINE" ]; then
        echo "  Latest: ${STEP_LINE}"
    else
        echo "  (no step info found in log)"
    fi
else
    echo "  No recent log files found."
fi

# Also check checkpoint directories for latest saved step
section "Latest Saved Checkpoints"
for ckpt_dir in "${TRAINING_DIR}"/checkpoints_v* "${TRAINING_DIR}"/value_checkpoints; do
    if [ -d "$ckpt_dir" ]; then
        latest=$(ls -d "${ckpt_dir}"/iter_* 2>/dev/null | sort -V | tail -1)
        if [ -n "$latest" ]; then
            echo "  $(basename "$ckpt_dir"): $(basename "$latest")"
        else
            echo "  $(basename "$ckpt_dir"): (no iterations saved)"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Latest Reward Metrics
# ---------------------------------------------------------------------------

section "Latest Reward Metrics"

if [ -n "$LATEST_LOG" ]; then
    # Look for reward-related lines
    REWARD_LINES=$(grep -i 'reward\|pass@\|accuracy\|success_rate' "$LATEST_LOG" 2>/dev/null | tail -5)
    if [ -n "$REWARD_LINES" ]; then
        echo "$REWARD_LINES" | while IFS= read -r line; do
            echo "  $line"
        done
    else
        echo "  No reward metrics found in latest log."
    fi
else
    echo "  No log files available."
fi

# ---------------------------------------------------------------------------
# Disk Usage
# ---------------------------------------------------------------------------

section "Disk Usage"

echo "  Checkpoints:"
for ckpt_dir in "${TRAINING_DIR}"/checkpoints_v* "${TRAINING_DIR}"/value_checkpoints; do
    if [ -d "$ckpt_dir" ]; then
        size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
        count=$(find "$ckpt_dir" -maxdepth 1 -name "iter_*" -type d 2>/dev/null | wc -l)
        echo "    $(basename "$ckpt_dir"): ${size} (${count} iterations)"
    fi
done

echo ""
echo "  Models:"
if [ -d "${WORKSPACE}/models" ]; then
    du -sh "${WORKSPACE}/models"/* 2>/dev/null | while read -r size path; do
        echo "    $(basename "$path"): ${size}"
    done
fi

echo ""
echo "  Data:"
if [ -d "${TRAINING_DIR}/data" ]; then
    du -sh "${TRAINING_DIR}/data" 2>/dev/null | while read -r size path; do
        echo "    ${size} total"
    done
fi

echo ""
echo "  Filesystem:"
df -h "$(dirname "${WORKSPACE}")" 2>/dev/null | tail -1 | \
    awk '{printf "    %s used / %s total (%s)\n", $3, $2, $5}'

echo ""
