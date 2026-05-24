"""
Co-Training Orchestrator.

Coordinates the full generator + value model co-training loop:

  Round N:
    1. Proof Search   -- Launch SGLang with current generator checkpoint,
                         run trajectory_collector.py to gather (state, label) pairs.
    2. Value Training  -- Train value model on ALL accumulated trajectories
                         via train_value_slime.sh (SLIME / Megatron-LM SFT).
    3. GRPO Training   -- Train generator with step-level GRPO
                         via train_step_grpo.sh.
    4. Evaluation      -- Run evaluate.py on MiniF2F to track progress.

Designed to run inside the SLIME Docker container where /workspace is mounted.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("co_train")

# ---------------------------------------------------------------------------
# Optional wandb import
# ---------------------------------------------------------------------------

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not installed -- metrics will only be printed to stdout")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoTrainConfig:
    """All tuneable knobs for the co-training loop."""

    # Top-level
    num_rounds: int = 5
    base_dir: str = "/workspace/training"
    start_round: int = 0

    # Checkpoint directories
    generator_checkpoint_dir: str = "/workspace/training/generator_checkpoints"
    value_checkpoint_dir: str = "/workspace/training/value_checkpoints"
    trajectory_dir: str = "/workspace/training/trajectories"

    # Initial model paths
    generator_model_hf: str = "/workspace/models/DeepSeek-Prover-V2-7B"
    value_model_hf: str = "/workspace/models/Llama-3.2-1B"

    # Scripts
    grpo_script: str = "/workspace/training/train_step_grpo.sh"
    value_script: str = "/workspace/training/train_value_slime.sh"
    trajectory_script: str = "/workspace/training/trajectory_collector.py"
    evaluate_script: str = "/workspace/training/evaluate.py"

    # Proof search settings
    theorems_path: str = (
        "/workspace/ReProver/data/leandojo_benchmark_4/random/train.json"
    )
    num_theorems_per_round: int = 5000
    num_sampled_tactics: int = 32
    search_timeout: int = 300
    max_expansions: int = 128

    # SGLang inference server
    sglang_port: int = 30000
    sglang_tp: int = 4
    sglang_startup_timeout: int = 300  # seconds to wait for SGLang to be ready

    # Evaluation
    eval_dataset: str = "/workspace/training/data/minif2f.json"
    eval_n_samples: int = 8
    kimina_url: str = "http://localhost:8000"

    # GRPO control
    grpo_start_round: int = 0  # first round that includes GRPO training

    # wandb
    wandb_project: str = "lean-co-training"
    wandb_group: str = "co-train"
    wandb_enabled: bool = True

    @property
    def sglang_url(self) -> str:
        return f"http://localhost:{self.sglang_port}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(msg: str, char: str = "=", width: int = 72) -> None:
    border = char * width
    log.info("\n%s\n  %s\n%s", border, msg, border)


def _run(
    cmd: str,
    desc: str,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> int:
    """Run a shell command, streaming output. Returns the exit code."""
    _banner(desc)
    log.info("$ %s", cmd)

    merged_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            env=merged_env,
            timeout=timeout,
        )
        if result.returncode != 0:
            log.warning("%s exited with code %d", desc, result.returncode)
        return result.returncode
    except subprocess.TimeoutExpired:
        log.error("%s timed out after %d seconds", desc, timeout)
        return -1
    except Exception as exc:
        log.error("%s failed with exception: %s", desc, exc)
        return -1


def _count_lines(path: str) -> int:
    """Count lines in a file without reading it all into memory."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def _resolve_generator_checkpoint(cfg: CoTrainConfig, round_num: int) -> str:
    """Return the HF model path for the generator at a given round.

    Round 0 uses the base model.  Subsequent rounds use the checkpoint
    produced by the prior round's GRPO phase (if it exists), otherwise
    falls back to the most recent available checkpoint.
    """
    if round_num == 0:
        return cfg.generator_model_hf

    for r in range(round_num - 1, -1, -1):
        ckpt = os.path.join(cfg.generator_checkpoint_dir, f"round_{r}")
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            return ckpt

    log.warning(
        "No generator checkpoint found for rounds 0..%d, falling back to base model",
        round_num - 1,
    )
    return cfg.generator_model_hf


def _resolve_value_checkpoint(cfg: CoTrainConfig, round_num: int) -> Optional[str]:
    """Return the most recent value model checkpoint, or None."""
    for r in range(round_num - 1, -1, -1):
        ckpt = os.path.join(cfg.value_checkpoint_dir, f"round_{r}")
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            return ckpt
    return None


# ---------------------------------------------------------------------------
# SGLang lifecycle
# ---------------------------------------------------------------------------


def _kill_sglang() -> None:
    """Best-effort kill of any running SGLang servers."""
    subprocess.run("pkill -9 -f sglang", shell=True, capture_output=True)
    time.sleep(2)


def _launch_sglang(
    model_path: str, cfg: CoTrainConfig
) -> Optional[subprocess.Popen]:
    """Start an SGLang server in the background.  Returns the Popen handle."""
    _kill_sglang()

    cmd = (
        f"python -m sglang.launch_server"
        f" --model-path {model_path}"
        f" --port {cfg.sglang_port}"
        f" --tp {cfg.sglang_tp}"
        f" --trust-remote-code"
        f" --disable-radix-cache"
        f" --mem-fraction-static 0.85"
    )
    log.info("Launching SGLang: %s", cmd)

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Wait until the /health endpoint responds
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + cfg.sglang_startup_timeout
    health_url = f"http://localhost:{cfg.sglang_port}/health"

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.error("SGLang process exited prematurely (code %d)", proc.returncode)
            return None
        try:
            urllib.request.urlopen(health_url, timeout=5)
            log.info("SGLang server is ready at %s", cfg.sglang_url)
            return proc
        except (urllib.error.URLError, OSError):
            time.sleep(5)

    log.error("SGLang did not become healthy within %d seconds", cfg.sglang_startup_timeout)
    _stop_sglang(proc)
    return None


def _stop_sglang(proc: Optional[subprocess.Popen]) -> None:
    """Gracefully stop an SGLang process group."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    _kill_sglang()


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase_proof_search(cfg: CoTrainConfig, round_num: int) -> Optional[str]:
    """Phase 1: Launch SGLang with current generator, collect trajectories.

    Returns the path to this round's trajectory JSONL, or None on failure.
    """
    _banner(f"ROUND {round_num} -- Phase 1: Proof Search & Trajectory Collection")

    output_path = os.path.join(cfg.trajectory_dir, f"round_{round_num}.jsonl")
    os.makedirs(cfg.trajectory_dir, exist_ok=True)

    generator_ckpt = _resolve_generator_checkpoint(cfg, round_num)
    log.info("Generator checkpoint: %s", generator_ckpt)

    # Launch SGLang with the current generator
    sglang_proc = _launch_sglang(generator_ckpt, cfg)
    if sglang_proc is None:
        log.error("Failed to start SGLang -- skipping proof search for round %d", round_num)
        return None

    try:
        cmd = (
            f"python {cfg.trajectory_script}"
            f" --theorems {cfg.theorems_path}"
            f" --sglang-url {cfg.sglang_url}"
            f" --output {output_path}"
            f" --num-theorems {cfg.num_theorems_per_round}"
            f" --num-sampled-tactics {cfg.num_sampled_tactics}"
            f" --timeout {cfg.search_timeout}"
            f" --max-expansions {cfg.max_expansions}"
        )
        rc = _run(cmd, f"Round {round_num}: trajectory_collector.py")
        if rc != 0:
            log.error("Trajectory collection failed (exit code %d)", rc)
            return None
    finally:
        _stop_sglang(sglang_proc)

    if not os.path.exists(output_path):
        log.error("Expected trajectory file does not exist: %s", output_path)
        return None

    n = _count_lines(output_path)
    log.info("Collected %d trajectory states -> %s", n, output_path)
    return output_path


def phase_aggregate_trajectories(
    cfg: CoTrainConfig, round_num: int
) -> str:
    """Concatenate trajectories from all rounds into a single file.

    Returns the path to the aggregated JSONL.
    """
    _banner(f"ROUND {round_num} -- Aggregating Trajectories (rounds 0..{round_num})")

    agg_path = os.path.join(cfg.trajectory_dir, "all.jsonl")

    total = 0
    with open(agg_path, "w") as out:
        for r in range(round_num + 1):
            rpath = os.path.join(cfg.trajectory_dir, f"round_{r}.jsonl")
            if not os.path.exists(rpath):
                log.warning("No trajectory file for round %d (%s)", r, rpath)
                continue
            with open(rpath) as f:
                for line in f:
                    out.write(line)
                    total += 1

    log.info("Aggregated %d total trajectory states -> %s", total, agg_path)
    return agg_path


def phase_train_value(cfg: CoTrainConfig, round_num: int) -> int:
    """Phase 2: Train value model on accumulated trajectories via SLIME.

    The training script (train_value_slime.sh) reads DATA_PATH and SAVE_DIR
    from environment overrides when provided.
    """
    _banner(f"ROUND {round_num} -- Phase 2: Value Model Training")

    agg_path = os.path.join(cfg.trajectory_dir, "all.jsonl")
    if not os.path.exists(agg_path) or _count_lines(agg_path) == 0:
        log.warning("No aggregated trajectories -- skipping value training")
        return -1

    save_dir = os.path.join(cfg.value_checkpoint_dir, f"round_{round_num}")
    os.makedirs(save_dir, exist_ok=True)

    env = {
        "DATA_PATH": agg_path,
        "SAVE_DIR": save_dir,
    }

    rc = _run(
        f"bash {cfg.value_script}",
        f"Round {round_num}: train_value_slime.sh",
        env=env,
    )
    return rc


def phase_train_generator(cfg: CoTrainConfig, round_num: int) -> int:
    """Phase 3: GRPO generator training via SLIME.

    The training script (train_step_grpo.sh) reads SAVE_DIR from the
    environment when provided.
    """
    _banner(f"ROUND {round_num} -- Phase 3: GRPO Generator Training")

    if round_num < cfg.grpo_start_round:
        log.info(
            "Skipping GRPO (grpo_start_round=%d, current round=%d)",
            cfg.grpo_start_round,
            round_num,
        )
        return 0

    save_dir = os.path.join(cfg.generator_checkpoint_dir, f"round_{round_num}")
    os.makedirs(save_dir, exist_ok=True)

    # If a previous generator checkpoint exists, use it as the starting point
    prev_ckpt = _resolve_generator_checkpoint(cfg, round_num)
    env = {
        "SAVE_DIR": save_dir,
    }
    # Pass previous checkpoint so the script can --load from it
    if prev_ckpt != cfg.generator_model_hf:
        env["LOAD_DIR"] = prev_ckpt

    rc = _run(
        f"bash {cfg.grpo_script}",
        f"Round {round_num}: train_step_grpo.sh",
        env=env,
    )
    return rc


def phase_evaluate(cfg: CoTrainConfig, round_num: int) -> Optional[Dict]:
    """Phase 4: Evaluate on MiniF2F (or configured dataset).

    Returns the results dict, or None on failure.
    """
    _banner(f"ROUND {round_num} -- Phase 4: Evaluation")

    if not os.path.exists(cfg.eval_dataset):
        log.warning("Eval dataset not found at %s -- skipping evaluation", cfg.eval_dataset)
        return None

    # Use the checkpoint just produced by this round (or latest available)
    generator_ckpt = _resolve_generator_checkpoint(cfg, round_num + 1)
    log.info("Evaluating generator: %s", generator_ckpt)

    sglang_proc = _launch_sglang(generator_ckpt, cfg)
    if sglang_proc is None:
        log.error("Failed to start SGLang for evaluation -- skipping")
        return None

    results_path = os.path.join(
        cfg.base_dir, "eval_results", f"round_{round_num}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    rc = -1
    try:
        cmd = (
            f"python {cfg.evaluate_script}"
            f" --dataset {cfg.eval_dataset}"
            f" --sglang-url {cfg.sglang_url}"
            f" --kimina-url {cfg.kimina_url}"
            f" --n-samples {cfg.eval_n_samples}"
            f" --output {results_path}"
        )
        rc = _run(cmd, f"Round {round_num}: evaluate.py")
    finally:
        _stop_sglang(sglang_proc)

    if rc != 0 or not os.path.exists(results_path):
        log.error("Evaluation failed or produced no output")
        return None

    with open(results_path) as f:
        results = json.load(f)

    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def co_train(cfg: CoTrainConfig) -> None:
    """Run the full co-training loop."""

    # Initialize wandb
    wb_run = None
    if cfg.wandb_enabled and WANDB_AVAILABLE:
        wb_run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            config={
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(cfg).items()
            },
            resume="allow",
        )
        log.info("wandb run: %s", wb_run.url)

    _banner(
        f"CO-TRAINING: {cfg.num_rounds} rounds "
        f"(starting from round {cfg.start_round})",
        char="#",
    )
    log.info("Config:\n%s", json.dumps(vars(cfg), indent=2))

    round_metrics: List[Dict] = []

    for round_num in range(cfg.start_round, cfg.start_round + cfg.num_rounds):
        round_start = time.time()
        _banner(f"CO-TRAINING ROUND {round_num}", char="#")

        metrics: Dict = {"round": round_num}

        # -- Phase 1: Proof search & trajectory collection -----------------
        try:
            traj_path = phase_proof_search(cfg, round_num)
            if traj_path:
                n_new = _count_lines(traj_path)
                metrics["trajectories_new"] = n_new
                log.info("Round %d: collected %d new trajectory states", round_num, n_new)
            else:
                metrics["trajectories_new"] = 0
                log.warning("Round %d: proof search produced no trajectories", round_num)
        except Exception:
            log.exception("Round %d: proof search phase failed", round_num)
            metrics["trajectories_new"] = 0

        # -- Aggregate trajectories ----------------------------------------
        try:
            agg_path = phase_aggregate_trajectories(cfg, round_num)
            metrics["trajectories_total"] = _count_lines(agg_path)
        except Exception:
            log.exception("Round %d: trajectory aggregation failed", round_num)
            metrics["trajectories_total"] = 0

        # -- Phase 2: Value model training ---------------------------------
        try:
            rc = phase_train_value(cfg, round_num)
            metrics["value_train_exit_code"] = rc
        except Exception:
            log.exception("Round %d: value training phase failed", round_num)
            metrics["value_train_exit_code"] = -1

        # -- Phase 3: GRPO generator training ------------------------------
        try:
            rc = phase_train_generator(cfg, round_num)
            metrics["grpo_train_exit_code"] = rc
        except Exception:
            log.exception("Round %d: GRPO training phase failed", round_num)
            metrics["grpo_train_exit_code"] = -1

        # -- Phase 4: Evaluation -------------------------------------------
        try:
            eval_results = phase_evaluate(cfg, round_num)
            if eval_results and "pass_at_k" in eval_results:
                for k, v in eval_results["pass_at_k"].items():
                    metrics[f"eval_pass_at_{k}"] = v
                total_problems = len(eval_results.get("results", {}))
                if total_problems > 0:
                    for k, v in eval_results["pass_at_k"].items():
                        metrics[f"eval_pass_at_{k}_rate"] = v / total_problems
        except Exception:
            log.exception("Round %d: evaluation phase failed", round_num)

        # -- Round summary -------------------------------------------------
        elapsed = time.time() - round_start
        metrics["round_duration_minutes"] = round(elapsed / 60, 1)
        round_metrics.append(metrics)

        _banner(f"ROUND {round_num} COMPLETE -- {elapsed/60:.1f} minutes")
        log.info("Round %d metrics: %s", round_num, json.dumps(metrics, indent=2))

        # Log to wandb
        if wb_run is not None:
            wb_run.log(metrics, step=round_num)

        # Persist metrics to disk so we can resume analysis later
        metrics_path = os.path.join(cfg.base_dir, "co_train_metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # -- Final summary -----------------------------------------------------
    _banner("CO-TRAINING COMPLETE", char="#")
    log.info(
        "Completed %d rounds.  Metrics saved to %s",
        cfg.num_rounds,
        os.path.join(cfg.base_dir, "co_train_metrics.jsonl"),
    )

    if wb_run is not None:
        wb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> CoTrainConfig:
    parser = argparse.ArgumentParser(
        description="Co-training orchestrator for Lean4 prover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rounds
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of co-training rounds")
    parser.add_argument("--start-round", type=int, default=0, help="Resume from this round number")
    parser.add_argument("--base-dir", default="/workspace/training", help="Root directory for all artifacts")

    # Checkpoints
    parser.add_argument("--generator-checkpoint-dir", default=None)
    parser.add_argument("--value-checkpoint-dir", default=None)
    parser.add_argument("--trajectory-dir", default=None)

    # Models
    parser.add_argument("--generator-model-hf", default="/workspace/models/DeepSeek-Prover-V2-7B")
    parser.add_argument("--value-model-hf", default="/workspace/models/Llama-3.2-1B")

    # Scripts
    parser.add_argument("--grpo-script", default=None)
    parser.add_argument("--value-script", default=None)

    # Proof search
    parser.add_argument("--theorems-path", default="/workspace/ReProver/data/leandojo_benchmark_4/random/train.json")
    parser.add_argument("--num-theorems-per-round", type=int, default=5000)
    parser.add_argument("--num-sampled-tactics", type=int, default=32)
    parser.add_argument("--search-timeout", type=int, default=300)
    parser.add_argument("--max-expansions", type=int, default=128)

    # SGLang
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--sglang-tp", type=int, default=4)

    # Evaluation
    parser.add_argument("--eval-dataset", default="/workspace/training/data/minif2f.json")
    parser.add_argument("--eval-n-samples", type=int, default=8)
    parser.add_argument("--kimina-url", default="http://localhost:8000")

    # GRPO
    parser.add_argument("--grpo-start-round", type=int, default=0, help="First round that includes GRPO training")

    # wandb
    parser.add_argument("--wandb-project", default="lean-co-training")
    parser.add_argument("--wandb-group", default="co-train")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Build config, applying defaults derived from base_dir
    cfg = CoTrainConfig(
        num_rounds=args.num_rounds,
        start_round=args.start_round,
        base_dir=args.base_dir,
        generator_checkpoint_dir=args.generator_checkpoint_dir or f"{args.base_dir}/generator_checkpoints",
        value_checkpoint_dir=args.value_checkpoint_dir or f"{args.base_dir}/value_checkpoints",
        trajectory_dir=args.trajectory_dir or f"{args.base_dir}/trajectories",
        generator_model_hf=args.generator_model_hf,
        value_model_hf=args.value_model_hf,
        grpo_script=args.grpo_script or f"{args.base_dir}/train_step_grpo.sh",
        value_script=args.value_script or f"{args.base_dir}/train_value_slime.sh",
        trajectory_script=f"{args.base_dir}/trajectory_collector.py",
        evaluate_script=f"{args.base_dir}/evaluate.py",
        theorems_path=args.theorems_path,
        num_theorems_per_round=args.num_theorems_per_round,
        num_sampled_tactics=args.num_sampled_tactics,
        search_timeout=args.search_timeout,
        max_expansions=args.max_expansions,
        sglang_port=args.sglang_port,
        sglang_tp=args.sglang_tp,
        eval_dataset=args.eval_dataset,
        eval_n_samples=args.eval_n_samples,
        kimina_url=args.kimina_url,
        grpo_start_round=args.grpo_start_round,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_enabled=not args.no_wandb,
    )

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    co_train(cfg)
