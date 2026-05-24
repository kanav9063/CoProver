# DeepSeek-Prover-V2-7B model config for Megatron
# Architecture: LlamaForCausalLM (from config.json)
# hidden_size=4096, num_layers=30, num_heads=32, intermediate_size=11008
# vocab_size=102400, rope_theta=10000, rms_norm_eps=1e-06

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
