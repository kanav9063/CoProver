#!/bin/bash
# Step-Level GRPO Training for Lean4 Tactic Generator
# Model: DeepSeek-Prover-V2-7B (cold start)
# Hardware: 8x H200 (colocated — all 8 shared between train + rollout)

set -ex

export PYTHONUNBUFFERED=1
export DISABLE_REMOTE_CACHE=1

# ===== Paths =====
MODEL_HF="/workspace/models/DeepSeek-Prover-V2-7B"
MODEL_MEGATRON="/workspace/models/DeepSeek-Prover-V2-7B_torch_dist"
SAVE_DIR="/workspace/training/checkpoints_v4"
DATA_PATH="/workspace/training/data/lean_tactics.jsonl"

export PATH="/root/.elan/bin:$PATH"
mkdir -p $SAVE_DIR

# ===== Model Config (DeepSeek-Prover-V2-7B = LlamaForCausalLM) =====
MODEL_ARGS=(
   --swiglu
   --num-layers 30
   --hidden-size 4096
   --ffn-hidden-size 11008
   --num-attention-heads 32
   --num-query-groups 32
   --max-position-embeddings 65536
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --vocab-size 102400
   --kv-channels 128
   --use-rope-scaling
   --rotary-scaling-factor 16.0
   --untie-embeddings-and-output-weights
)

CKPT_ARGS=(
   --hf-checkpoint $MODEL_HF
   --ref-load $MODEL_MEGATRON
   --load $SAVE_DIR
   --save $SAVE_DIR
   --save-interval 50
)

# ===== Rollout — matching working config =====
ROLLOUT_ARGS=(
   --prompt-data /workspace/training/data/lean_whole_proof.jsonl
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle

   --custom-rm-path training.lean_reward.compute_reward

   # 64 prompts × 8 samples = 512 per rollout
   --num-rollout 500
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --global-batch-size 512

   --rollout-max-response-len 4096
   --rollout-temperature 1

   --balance-data
)

# ===== Performance — colocate all 8 GPUs =====
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

# ===== GRPO — matching working config =====
GRPO_ARGS=(
   --advantage-estimator grpo
   --calculate-per-token-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-disable-radix-cache
   --sglang-max-running-requests 32
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project lean-step-grpo
   --wandb-group deepseek-prover-v2-7b
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ===== Debug (all commented out for full training) =====
DEBUG_ARGS=(
)

# ===== Launch =====
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/workspace\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PATH\": \"/root/.elan/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\",
    \"WANDB_API_KEY\": \"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\",
    \"DISABLE_REMOTE_CACHE\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /root/slime \
   -- python3 /root/slime/train.py \
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
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DEBUG_ARGS[@]}
