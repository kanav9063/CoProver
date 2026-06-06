# Makefile — Common targets for the Lean formal proof co-training pipeline
#
# Most training commands run inside Docker (slimerl/slime:latest).
# The kimina-lean-server runs in its own container on port 8000.
#
# Override defaults via environment:
#   DOCKER_IMAGE=slimerl/slime:latest  WORKSPACE=/mnt/filesystem-m5/formal  make train-generator

SHELL := /bin/bash
.DEFAULT_GOAL := help

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
HOST_WORKSPACE := $(if $(filter training,$(notdir $(SCRIPT_DIR))),$(abspath $(SCRIPT_DIR)/..),$(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE       ?= /mnt/filesystem-m5/formal
DOCKER_IMAGE    ?= slimerl/slime:latest
KIMINA_IMAGE    ?= kimina-lean-server:latest
KIMINA_PORT     ?= 8000
SGLANG_GEN_PORT ?= 30000
SGLANG_VAL_PORT ?= 30001
TRAINING_DIR    := $(WORKSPACE)/training
DOCKER_BIN      := $(strip $(shell if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then echo docker; elif command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then echo "sudo docker"; fi))

DOCKER_RUN := $(DOCKER_BIN) run --rm \
	--gpus all \
	--ipc=host \
	--shm-size=16g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--add-host=host.docker.internal:host-gateway \
	-v $(WORKSPACE):/workspace \
	-e PYTHONUNBUFFERED=1 \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-w /workspace/training

define require_docker
	@if [ -z "$(DOCKER_BIN)" ]; then \
		echo "Docker is unavailable. Install Docker or configure passwordless sudo for docker." >&2; \
		exit 1; \
	fi
endef

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: help preflight setup data convert train-generator train-value collect \
        eval co-train status kill clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Setup -----------------------------------------------------------------

preflight: ## Check Docker, Python modules, workspace paths, and model dirs
	@WORKSPACE="$(HOST_WORKSPACE)" TRAINING_DIR="$(SCRIPT_DIR)" bash $(SCRIPT_DIR)/check_setup.sh

setup: ## Pull Docker images and start kimina-lean-server
	$(require_docker)
	@echo "==> Pulling Docker images ..."
	$(DOCKER_BIN) pull $(DOCKER_IMAGE)
	$(DOCKER_BIN) pull $(KIMINA_IMAGE) 2>/dev/null || \
		echo "WARNING: $(KIMINA_IMAGE) not found in registry; ensure it is built locally."
	@echo "==> Starting servers ..."
	WORKSPACE="$(HOST_WORKSPACE)" TRAINING_DIR="$(SCRIPT_DIR)" bash $(SCRIPT_DIR)/launch_servers.sh --kimina-only

# --- Data ------------------------------------------------------------------

data: ## Download and prepare all datasets (runs inside Docker)
	$(require_docker)
	$(DOCKER_RUN) $(DOCKER_IMAGE) \
		bash /workspace/training/prepare_all.sh

# --- Conversion ------------------------------------------------------------

convert: ## Convert HuggingFace weights to Megatron torch_dist format
	$(require_docker)
	@echo "==> Converting DeepSeek-Prover-V2-7B to Megatron format ..."
	$(DOCKER_RUN) $(DOCKER_IMAGE) \
		bash -c "cd /root/slime && PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
			--input-dir /workspace/models/DeepSeek-Prover-V2-7B \
			--output-dir /workspace/models/DeepSeek-Prover-V2-7B_torch_dist"
	@echo "==> Converting Llama-3.2-1B to Megatron format ..."
	$(DOCKER_RUN) $(DOCKER_IMAGE) \
		bash -c "cd /root/slime && PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
			--input-dir /workspace/models/Llama-3.2-1B \
			--output-dir /workspace/models/Llama-3.2-1B_torch_dist"

# --- Training --------------------------------------------------------------

train-generator: ## Run GRPO training for the tactic generator
	$(require_docker)
	$(DOCKER_RUN) -it $(DOCKER_IMAGE) \
		bash /workspace/training/train_step_grpo.sh

train-value: ## Run value model SFT training
	$(require_docker)
	$(DOCKER_RUN) -it $(DOCKER_IMAGE) \
		bash /workspace/training/train_value_slime.sh

# --- Trajectory collection -------------------------------------------------

collect: ## Run trajectory collection using the current generator
	$(require_docker)
	$(DOCKER_RUN) $(DOCKER_IMAGE) \
		python /workspace/training/trajectory_collector.py \
			--sglang-url http://host.docker.internal:$(SGLANG_GEN_PORT) \
			--kimina-url http://host.docker.internal:$(KIMINA_PORT) \
			--output /workspace/training/data/trajectories.jsonl

# --- Evaluation ------------------------------------------------------------

eval: ## Run MiniF2F evaluation against the current generator
	$(require_docker)
	$(DOCKER_RUN) $(DOCKER_IMAGE) \
		python /workspace/training/evaluate.py \
			--sglang-url http://host.docker.internal:$(SGLANG_GEN_PORT) \
			--kimina-url http://host.docker.internal:$(KIMINA_PORT) \
			--data /workspace/training/data/minif2f_test.jsonl

# --- Co-Training -----------------------------------------------------------

co-train: ## Run the full co-training loop (generator + value, multiple rounds)
	bash $(TRAINING_DIR)/co_train.sh $(CO_TRAIN_ARGS)

# --- Operations ------------------------------------------------------------

status: ## Check training status (GPUs, servers, latest metrics, disk)
	@WORKSPACE="$(HOST_WORKSPACE)" TRAINING_DIR="$(SCRIPT_DIR)" bash $(SCRIPT_DIR)/check_status.sh

kill: ## Kill all training processes (ray, sglang, docker training containers)
	$(require_docker)
	@echo "==> Stopping ray ..."
	-ray stop --force 2>/dev/null
	-pkill -9 ray 2>/dev/null
	@echo "==> Stopping sglang ..."
	-pkill -9 sglang 2>/dev/null
	@echo "==> Stopping training containers ..."
	-$(DOCKER_BIN) ps -q --filter ancestor=$(DOCKER_IMAGE) | xargs -r $(DOCKER_BIN) kill
	@echo "==> Done. All training processes killed."

clean: ## Remove checkpoints and logs (asks for confirmation)
	@echo "This will remove ALL checkpoint directories and log files under:"
	@echo "  $(SCRIPT_DIR)/checkpoints*"
	@echo "  $(SCRIPT_DIR)/value_checkpoints"
	@echo ""
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || { echo "Aborted."; exit 1; }
	rm -rf $(SCRIPT_DIR)/checkpoints_v*/
	rm -rf $(SCRIPT_DIR)/value_checkpoints/
	rm -f $(SCRIPT_DIR)/*.log
	@echo "Cleaned."
