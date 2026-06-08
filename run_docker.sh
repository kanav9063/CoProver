#!/usr/bin/env bash
# Launch the SLIME docker container with the workspace mounted at /workspace.
# The helper mirrors Makefile networking so host-side prover/model servers remain
# reachable from inside the container via host.docker.internal.
# Override defaults with WORKSPACE=/path DOCKER_IMAGE=image[:tag] bash run_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${TRAINING_DIR:-}" ]; then
  TRAINING_DIR="${TRAINING_DIR}"
elif [ -n "${WORKSPACE:-}" ] && [ -f "${WORKSPACE}/training/run_docker.sh" ]; then
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

DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:latest}"

die() {
  echo "run_docker.sh: $*" >&2
  exit 1
}

if [ ! -d "${WORKSPACE}" ]; then
  die "workspace does not exist: ${WORKSPACE}"
fi

if [ ! -d "${TRAINING_DIR}" ]; then
  die "training directory does not exist: ${TRAINING_DIR}"
fi

if [ ! -f "${TRAINING_DIR}/run_docker.sh" ]; then
  die "training helper not found under: ${TRAINING_DIR}"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_CMD=(docker)
elif command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
  DOCKER_CMD=(sudo docker)
else
  die "docker is unavailable; install Docker or configure passwordless sudo for docker"
fi

echo "Launching ${DOCKER_IMAGE} with WORKSPACE=${WORKSPACE}"

"${DOCKER_CMD[@]}" run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --add-host=host.docker.internal:host-gateway \
  -v "${WORKSPACE}:/workspace" \
  -w /workspace \
  -it "${DOCKER_IMAGE}" \
  /bin/bash
