<p align="center">
  <img src="https://img.shields.io/badge/Lean-4-blue?style=for-the-badge" alt="Lean 4"/>
  <img src="https://img.shields.io/badge/GRPO-RL-red?style=for-the-badge" alt="GRPO"/>
  <img src="https://img.shields.io/badge/Co--Training-Value+Generator-green?style=for-the-badge" alt="Co-Training"/>
  <img src="https://img.shields.io/badge/H200-8xGPU-purple?style=for-the-badge" alt="H200"/>
</p>

<h1 align="center">CoProver</h1>

<p align="center">
  <b>Co-Training Proof Generation and Search Heuristics for Lean 4 Theorem Proving</b>
</p>

<p align="center">
  <i>Learning not only what tactic to try, but which proof state to explore next.</i>
</p>

> **CONFIDENTIAL** — This repository contains proprietary research code. Trained models will be released on Hugging Face soon.

---

## Overview

Neural theorem provers have improved dramatically — DeepSeek-Prover-V2 achieves 88.9% on MiniF2F, Goedel-Prover-V2 reaches 90.4%. But they all share one asymmetry: **the tactic generator is trained, the search strategy is not.**

CoProver fixes this. We co-train two models:

| | **Tactic Generator** | **Value Model** |
|---|---|---|
| **Model** | DeepSeek-Prover-V2-7B | Llama-3.2-1B |
| **Task** | Generate complete Lean 4 proofs | Score proof states by distance to QED |
| **Training** | GRPO (Lean verification reward) | SFT on proof trajectories ($\gamma^d$ labels) |
| **Output** | Lean 4 code | Scalar $V(s) \in [0,1]$ |

The value model predicts $\gamma^d$ — a discounted distance to proof completion. States one step from QED score ~0.95, states five steps away score ~0.77, dead ends score 0.0. This replaces hand-designed heuristics (cumulative log-probability) with learned ones.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              CoProver System                 │
                    │                                             │
                    │  ┌───────────────────┐  ┌───────────────┐  │
                    │  │  Tactic Generator  │  │  Value Model  │  │
                    │  │  DeepSeek-V2 7B    │  │  Llama 1B     │  │
                    │  │                    │  │               │  │
                    │  │  Input: theorem    │  │  Input: state │  │
                    │  │  Output: proof     │  │  Output: γ^d  │  │
                    │  └────────┬───────────┘  └───────┬───────┘  │
                    │           │                       │          │
                    │           ▼                       ▼          │
                    │  ┌─────────────────────────────────────────┐│
                    │  │        Value-Guided Best-First Search   ││
                    │  │                                         ││
                    │  │  priority = α·logprob + (1-α)·V(state) ││
                    │  │                                         ││
                    │  │  1. Pop highest-value state             ││
                    │  │  2. Generator proposes K tactics        ││
                    │  │  3. Lean verifies each tactic           ││
                    │  │  4. Value model scores new states       ││
                    │  │  5. Push scored states to queue         ││
                    │  │  6. Repeat until QED or budget          ││
                    │  └─────────────────┬───────────────────────┘│
                    │                    │                         │
                    │                    ▼                         │
                    │  ┌─────────────────────────────────────────┐│
                    │  │         Kimina Lean Server              ││
                    │  │    Parallel Lean 4 Proof Checking       ││
                    │  │    POST /api/check → valid / invalid    ││
                    │  └─────────────────────────────────────────┘│
                    └─────────────────────────────────────────────┘
```

## Co-Training Loop

The two models improve each other through iterative co-training:

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                    CO-TRAINING ROUND N                       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  Phase 1: PROOF SEARCH                                       ║
  ║  ┌──────────────────────────────────────────────────────┐   ║
  ║  │ Generator attempts proofs on 5,000 theorems          │   ║
  ║  │ Value model guides search (if available)              │   ║
  ║  │ Lean verifies each attempt                            │   ║
  ║  │ → Collect trajectories with γ^d labels                │   ║
  ║  └──────────────────────────────────────────────────────┘   ║
  ║                          │                                   ║
  ║                          ▼                                   ║
  ║  Phase 2: TRAIN VALUE MODEL                                  ║
  ║  ┌──────────────────────────────────────────────────────┐   ║
  ║  │ SFT on ALL accumulated trajectories (rounds 0..N)     │   ║
  ║  │ Positive: γ^d where d = steps remaining to QED        │   ║
  ║  │ Negative: 0.0 (failed proof paths)                    │   ║
  ║  └──────────────────────────────────────────────────────┘   ║
  ║                          │                                   ║
  ║                          ▼                                   ║
  ║  Phase 3: TRAIN GENERATOR (GRPO)                             ║
  ║  ┌──────────────────────────────────────────────────────┐   ║
  ║  │ Whole-proof GRPO: generate proof → Lean verifies      │   ║
  ║  │ Reward = 1.0 (verified) or 0.0 (failed)              │   ║
  ║  │ 64 prompts × 8 samples = 512 per rollout             │   ║
  ║  └──────────────────────────────────────────────────────┘   ║
  ║                          │                                   ║
  ║                          ▼                                   ║
  ║  Phase 4: EVALUATE                                           ║
  ║  ┌──────────────────────────────────────────────────────┐   ║
  ║  │ pass@1, pass@8, pass@32 on MiniF2F / PutnamBench     │   ║
  ║  │ Search efficiency: nodes expanded per proof           │   ║
  ║  └──────────────────────────────────────────────────────┘   ║
  ║                                                              ║
  ║  Better search → more proofs → better data → better models  ║
  ╚══════════════════════════════════════════════════════════════╝
```

### Why Co-Training Matters

Training the models independently works, but co-training compounds their improvements:

```
  Independent Training:          Co-Training:

  Generator v1 ──► eval          Generator v1 ──► search ──► trajectories
  Value v1     ──► eval              │                          │
                                     ▼                          ▼
  Generator v2 ──► eval          Value v1 trained ◄──── from round 1 data
  Value v2     ──► eval              │
                                     ▼
  (no interaction)               Generator v2 ──► search with Value v1
                                     │                          │
                                     ▼                          ▼
                                 Value v2 trained ◄──── from round 1+2 data
                                     │
                                     ▼
                                 (models adapt to each other)
```

The value model must track the generator — a value model trained on Generator v1's trajectories gives poor scores for Generator v3's states. Co-training keeps them calibrated.

## The Value Model: γ^d Scoring

Unlike binary (proved/not proved) labels, we use **discounted distance** to give the value model a gradient of information:

```
  Proof trace:  State₀ → State₁ → State₂ → State₃ → QED
                  │         │         │         │
  Label:        γ⁴=0.81  γ³=0.86  γ²=0.90  γ¹=0.95   (γ=0.95)

  Failed trace: State₀ → State₁ → State₂ → ERROR
                  │         │         │
  Label:        0.00      0.00      0.00
```

This tells the value model not just *whether* a state leads to a proof, but *how close* it is. A state two steps from QED (label 0.90) is more valuable than one five steps away (label 0.77).

## Repository Structure

```
coprover/
├── README.md
├── Makefile                      # make train-generator, make eval, make co-train
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── generator_grpo.yaml       # GRPO hyperparameters
│   ├── value_model_sft.yaml      # Value model training config
│   └── co_training.yaml          # Full co-training loop config
│
├── models/
│   └── deepseek-prover-v2-7B.sh  # Megatron model architecture args
│
│── ── Training ──────────────────────────────────────────────
│
├── train_step_grpo.sh            # SLIME GRPO for generator
├── train_value_slime.sh          # SLIME SFT for value model
├── train_value_model.py          # HF Trainer alternative
├── lean_reward.py                # Reward: kimina verification → 1.0/0.0
├── value_reward.py               # Combined: Lean + value model bonus
│
│── ── Search & Inference ────────────────────────────────────
│
├── value_model.py                # Llama-3.2-1B + MLP value head → sigmoid
├── value_guided_search.py        # Best-first search with V(s) scoring
├── value_rollout.py              # Custom SLIME rollout with value guidance
├── lean_generate.py              # Single-tactic generation rollout
│
│── ── Data Pipeline ─────────────────────────────────────────
│
├── trajectory_collector.py       # Generate → verify → record γ^d labels
├── prepare_minif2f.py            # Download MiniF2F + Kimina promptset
├── prepare_value_data.py         # Trajectories → SFT format (with oversampling)
├── prepare_all.sh                # One-shot data preparation
├── filter_dataset.py             # Frontier difficulty filtering
│
│── ── Evaluation & Analysis ─────────────────────────────────
│
├── evaluate.py                   # pass@k with unbiased estimator
├── compare_checkpoints.py        # A/B model comparison
├── analyze_training.py           # Parse SLIME logs, plot curves
│
│── ── Orchestration ─────────────────────────────────────────
│
├── co_train.py                   # Multi-round co-training orchestrator
├── co_train.sh                   # Docker wrapper with server lifecycle
├── launch_servers.sh             # Start kimina + SGLang with health checks
├── convert_and_serve.sh          # Megatron → HuggingFace → SGLang
├── check_status.sh               # GPU, server health, latest metrics
├── run_docker.sh                 # Launch SLIME container
│
│── ── Infrastructure ────────────────────────────────────────
│
├── lean_server.py                # LeanDojo verification HTTP server
└── lean_server_pool.py           # Persistent Lean REPL pool with LRU cache
```

## Quick Start

```bash
# 1. Pull images
docker pull slimerl/slime:latest
docker pull projectnumina/kimina-lean-server:2.0.0

# 2. Start Lean verification server
docker run -d --name kimina-lean-server \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 --restart unless-stopped \
  projectnumina/kimina-lean-server:2.0.0

# 3. Enter SLIME container
bash run_docker.sh

# 4. Convert model weights to Megatron format
source models/deepseek-prover-v2-7B.sh
PYTHONPATH=/root/Megatron-LM python /root/slime/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /workspace/models/DeepSeek-Prover-V2-7B \
  --save /workspace/models/DeepSeek-Prover-V2-7B_torch_dist

# 5. Train the generator
bash train_step_grpo.sh

# 6. Run full co-training loop
bash co_train.sh --num-rounds 5
```

## Training Configurations

### Generator GRPO

| Parameter | Value |
|-----------|-------|
| Base model | DeepSeek-Prover-V2-7B (30 layers, 4096 hidden) |
| Learning rate | 1×10⁻⁶ (constant) |
| Batch size | 512 (64 prompts × 8 samples) |
| Reward | Binary Lean verification |
| Clipping | ε_low=0.2, ε_high=0.28 |
| Temperature | 1.0 |
| Max tokens | 4,096 |
| GPUs | 4 train (TP=2) + 4 rollout (SGLang) |
| Framework | SLIME (Megatron-LM + SGLang) |

### Value Model SFT

| Parameter | Value |
|-----------|-------|
| Base model | Llama-3.2-1B (16 layers, 2048 hidden) |
| Learning rate | 2×10⁻⁵ (cosine with warmup) |
| Labels | γ^d where d = steps to QED, γ=0.95 |
| Positive oversampling | 3× |
| Epochs | 3 per round |
| GPUs | 1 |

## Key Design Decisions

**Why whole-proof GRPO over step-level?** DeepSeek-Prover-V2-7B generates complete proofs naturally. Step-level GRPO requires a separate Lean REPL session per tactic, which is 100× slower than whole-proof verification via kimina-lean-server.

**Why no KL penalty?** With only 3-8% reward rate on hard theorems, KL regularization overly constrains exploration. The model needs freedom to discover novel proof strategies. We tried KL coefficients of 0.001 and 0.0001 — both stagnated training.

**Why γ^d over binary labels?** Binary labels (proved/not proved) throw away information about how *close* a failed attempt was. γ^d preserves the gradient — the value model learns that "3 steps from QED" is better than "10 steps away", even when neither attempt completed.

**Why Llama-3.2-1B for value model?** Small enough for fast inference during search (must score every expanded state), large enough to understand proof state structure. Served on SGLang alongside the generator with negligible overhead.

## Evaluation

```bash
# MiniF2F (244 problems)
python evaluate.py \
  --dataset data/minif2f_test.jsonl \
  --sglang-url http://localhost:30000 \
  --kimina-url http://localhost:8000 \
  --n-samples 32 --output eval_results/minif2f.json

# Compare two checkpoints
python compare_checkpoints.py \
  --sglang-url-a http://localhost:30000 \
  --sglang-url-b http://localhost:30001 \
  --label-a "baseline" --label-b "co-trained" \
  --dataset data/minif2f_test.jsonl \
  --n-samples 32 --output eval_results/comparison.json
```

## Related Work

| System | Generator Training | Search | Value/Critic | Co-Training |
|--------|-------------------|--------|--------------|-------------|
| DeepSeek-Prover-V2 | GRPO (whole-proof) | Fixed tree search | No | No |
| InternLM2.5-StepProver | Expert iteration | Critic-guided BFS | Yes (separate) | No (one round) |
| Kimina-Prover | GRPO (whole-proof) | Fixed | No | No |
| BFS-Prover | Expert iteration | Fixed BFS | No | No |
| AlphaProof | RL (unpublished) | MCTS | Yes (hand-tuned) | Unclear |
| **CoProver (ours)** | **GRPO** | **Learned value-guided** | **Yes (co-trained)** | **Yes (iterative)** |

## Hardware

- **8× NVIDIA H200** (140 GB VRAM each)
- 4 GPUs for Megatron-LM training (tensor parallelism 2)
- 4 GPUs for SGLang rollout generation
- kimina-lean-server on host CPU (Docker)
- ~3 min per GRPO training step (512 samples)

## Citation

```bibtex
@misc{coprover2026,
  title={CoProver: Co-Training Proof Generation and Search Heuristics for Automated Theorem Proving},
  year={2026},
  howpublished={\url{https://github.com/coprover}},
}
```

## License

Proprietary. All rights reserved. Models will be released on Hugging Face.
