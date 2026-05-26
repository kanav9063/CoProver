#!/usr/bin/env bash
# Launch the SLIME docker container with the workspace mounted at /workspace.
# Override defaults with WORKSPACE=/path DOCKER_IMAGE=image[:tag] bash run_docker.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/mnt/filesystem-m5/formal}"
DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:latest}"

die() {
  echo "run_docker.sh: $*" >&2
  exit 1
}

if [ ! -d "${WORKSPACE}" ]; then
  die "workspace does not exist: ${WORKSPACE}"
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
  -v "${WORKSPACE}:/workspace" \
  -w /workspace \
  -it "${DOCKER_IMAGE}" \
  /bin/bash
