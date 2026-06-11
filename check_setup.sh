#!/usr/bin/env bash
# check_setup.sh -- quick preflight checks for local prover tooling.
#
# Usage:
#   bash check_setup.sh
#   WORKSPACE=/path/to/formal bash check_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${TRAINING_DIR:-}" ]; then
  TRAINING_DIR="${TRAINING_DIR}"
elif [ -n "${WORKSPACE:-}" ] && [ -f "${WORKSPACE}/training/check_setup.sh" ]; then
  TRAINING_DIR="${WORKSPACE}/training"
else
  TRAINING_DIR="${SCRIPT_DIR}"
fi

if [ -n "${WORKSPACE:-}" ]; then
  WORKSPACE="${WORKSPACE}"
elif [ "$(basename "${TRAINING_DIR}")" = "training" ]; then
  WORKSPACE="$(cd "${TRAINING_DIR}/.." && pwd)"
else
  WORKSPACE="${TRAINING_DIR}"
fi

GENERATOR_MODEL="${GENERATOR_MODEL:-${WORKSPACE}/models/DeepSeek-Prover-V2-7B}"
VALUE_MODEL="${VALUE_MODEL:-${WORKSPACE}/models/Llama-3.2-1B}"
KIMINA_IMAGE="${KIMINA_IMAGE:-kimina-lean-server:latest}"
REQUIREMENTS_FILE="${TRAINING_DIR}/requirements.txt"

DOCKER_CMD=()
PYTHON_BIN=""
FAILURES=0

log() {
  printf '[preflight] %s\n' "$*"
}

pass() {
  log "PASS: $*"
}

warn() {
  log "WARN: $*"
}

fail() {
  log "FAIL: $*"
  FAILURES=$((FAILURES + 1))
}

find_python() {
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    return 0
  fi

  fail "python interpreter not found; install python3 or add python to PATH"
  return 1
}

find_docker() {
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return 0
  fi

  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
    return 0
  fi

  fail "docker is unavailable; install Docker or configure passwordless sudo for docker"
  return 1
}

check_path() {
  local label="$1"
  local path="$2"
  if [ -e "${path}" ]; then
    pass "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_command() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    pass "command available: ${cmd}"
  else
    fail "required command missing from PATH: ${cmd}"
  fi
}

check_any_command() {
  local description="$1"
  shift

  local cmd
  for cmd in "$@"; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      pass "${description}: ${cmd}"
      return 0
    fi
  done

  fail "${description} missing from PATH; install one of: $*"
}

check_python_module() {
  local module="$1"
  local package_hint="${2:-$1}"
  if "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1; then
    pass "python module '${module}' is importable via ${PYTHON_BIN}"
  else
    if [ -f "${REQUIREMENTS_FILE}" ]; then
      fail "python module '${module}' is missing for ${PYTHON_BIN}; install ${package_hint} via ${PYTHON_BIN} -m pip install -r ${REQUIREMENTS_FILE}"
    else
      fail "python module '${module}' is missing for ${PYTHON_BIN}; install package '${package_hint}'"
    fi
  fi
}

main() {
  log "Checking prover environment with WORKSPACE=${WORKSPACE}"

  find_docker && pass "docker is reachable via '${DOCKER_CMD[*]}'"
  find_python

  check_path "workspace" "${WORKSPACE}"
  check_path "training dir" "${TRAINING_DIR}"
  check_path "launch helper" "${TRAINING_DIR}/launch_servers.sh"
  check_path "generator model dir" "${GENERATOR_MODEL}"
  check_path "value model dir" "${VALUE_MODEL}"
  check_command curl
  check_command grep
  check_command nohup
  check_any_command "port inspection helper for early launcher diagnostics" lsof netstat

  if [ -n "${PYTHON_BIN}" ]; then
    pass "python interpreter: $(command -v "${PYTHON_BIN}")"
    check_path "requirements file" "${REQUIREMENTS_FILE}"
    check_python_module requests
    check_python_module aiohttp
    check_python_module datasets
    check_python_module matplotlib
    check_python_module numpy
    check_python_module sglang
    check_python_module torch
    check_python_module transformers
    check_python_module wandb
    check_python_module loguru
    check_python_module lean_dojo lean-dojo
  fi

  if [ ${#DOCKER_CMD[@]} -gt 0 ]; then
    if "${DOCKER_CMD[@]}" image inspect "${KIMINA_IMAGE}" >/dev/null 2>&1; then
      pass "kimina image available locally: ${KIMINA_IMAGE}"
    else
      warn "kimina image not present locally: ${KIMINA_IMAGE}"
    fi
  fi

  if [ -z "${WANDB_API_KEY:-}" ]; then
    warn "WANDB_API_KEY is unset; training launched from Docker will not log to Weights & Biases"
  else
    pass "WANDB_API_KEY is set"
  fi

  if [ "${FAILURES}" -gt 0 ]; then
    log "Preflight failed with ${FAILURES} issue(s)."
    exit 1
  fi

  log "Preflight passed."
}

main "$@"
