#!/bin/bash
# Value Model Training via SLIME (SFT mode)
# Model: Llama-3.2-1B
# Loss: sft_loss on (proof_state → score) pairs
# Data: trajectory JSONL collected from generator search

set -ex

export PYTHONUNBUFFERED=1

MODEL_HF="/workspace/models/Llama-3.2-1B"
MODEL_MEGATRON="/workspace/models/Llama-3.2-1B_torch_dist"
SAVE_DIR="/workspace/training/value_checkpoints"
DATA_PATH="/workspace/training/trajectories/trajectories_all.jsonl"

mkdir -p $SAVE_DIR

# ===== Model Config (Llama-3.2-1B) =====
MODEL_ARGS=(
   --swiglu
   --num-layers 16
   --hidden-size 2048
   --ffn-hidden-size 8192
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --max-position-embeddings 131072
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base 500000
   --vocab-size 128256
   --kv-channels 64
   --untie-embeddings-and-output-weights
)

CKPT_ARGS=(
   --hf-checkpoint $MODEL_HF
   --ref-load $MODEL_MEGATRON
   --load $SAVE_DIR
   --save $SAVE_DIR
   --save-interval 50
)

# ===== Data =====
ROLLOUT_ARGS=(
   --prompt-data $DATA_PATH
   --input-key prompt
   --label-key label
   --rollout-shuffle

   # SFT — no reward model needed
   --num-rollout 1
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --global-batch-size 64

   --rollout-max-response-len 32
   --rollout-temperature 0.0
)

# ===== SFT Loss =====
LOSS_ARGS=(
   --loss-type sft_loss
   --disable-compute-advantages-and-returns
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style cosine
   --lr-warmup-iters 10
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project lean-value-model
   --wandb-group llama-3.2-1b
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
    \"WANDB_API_KEY\": \"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\"
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
   ${LOSS_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]}
