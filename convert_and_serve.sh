#!/bin/bash
# Convert a SLIME/Megatron checkpoint back to HuggingFace format
# and serve it on SGLang for evaluation or next round of co-training.
#
# Usage: bash convert_and_serve.sh <checkpoint_dir> <output_hf_dir> <port>
# Example: bash convert_and_serve.sh checkpoints_v4/iter_0000049 models/generator_v4_step49 30000

set -ex

CKPT_DIR=${1:?Usage: convert_and_serve.sh <checkpoint_dir> <output_hf_dir> <port>}
OUTPUT_DIR=${2:?Usage: convert_and_serve.sh <checkpoint_dir> <output_hf_dir> <port>}
PORT=${3:-30000}

SLIME_DIR="/root/slime"
HF_ORIGINAL="/workspace/models/DeepSeek-Prover-V2-7B"

echo "Converting ${CKPT_DIR} -> ${OUTPUT_DIR}"

# Convert Megatron torch_dist -> HuggingFace
cd ${SLIME_DIR}
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir ${CKPT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --origin-hf-dir ${HF_ORIGINAL}

echo "Conversion done. Serving on port ${PORT}"

# Launch SGLang
python -m sglang.launch_server \
    --model-path ${OUTPUT_DIR} \
    --port ${PORT} \
    --dp 8
