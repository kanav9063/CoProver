#!/bin/bash
# GRPO training for Lean4 tactic generator using SLIME
# Model: DeepSeek-Prover-V2-7B (LlamaForCausalLM)
# Hardware: 8x H200 (4 train, 4 rollout)

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="/mnt/filesystem-m5/formal/slime"
MODEL_HF="/mnt/filesystem-m5/formal/models/DeepSeek-Prover-V2-7B"
MODEL_MEGATRON="/mnt/filesystem-m5/formal/models/DeepSeek-Prover-V2-7B_torch_dist"
SAVE_DIR="/mnt/filesystem-m5/formal/training/checkpoints"
DATA_PATH="/mnt/filesystem-m5/formal/training/data/lean_tactics.jsonl"

mkdir -p $SAVE_DIR

# ===== Step 0: Convert HF weights to Megatron format (if not done) =====
if [ ! -d "$MODEL_MEGATRON" ]; then
    echo "Converting HF weights to Megatron format..."
    cd $SLIME_DIR
    source "${SCRIPT_DIR}/models/deepseek-prover-v2-7B.sh"
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint $MODEL_HF \
        --save $MODEL_MEGATRON
    echo "Conversion complete."
fi

# ===== Model config =====
source "${SCRIPT_DIR}/models/deepseek-prover-v2-7B.sh"

CKPT_ARGS=(
   --hf-checkpoint $MODEL_HF
   --ref-load $MODEL_MEGATRON
   --load $SAVE_DIR
   --save $SAVE_DIR
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data $DATA_PATH
   --input-key prompt
   --label-key label
   --metadata-key metadata
   # No chat template — we use raw [GOAL]...[PROOFSTEP] format
   --rollout-shuffle

   # Custom Lean reward function
   --custom-rm-path training.lean_reward.compute_reward

   # Rollout sizing: 32 prompts × 4 samples = 128 per rollout
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 256
   --rollout-temperature 0.8

   --global-batch-size 128
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 4 rollout GPUs / 1 per engine = 4 SGLang engines (DP=4)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ===== Launch =====
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Kill any existing ray/sglang
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/mnt/filesystem-m5/formal\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

cd $SLIME_DIR

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
