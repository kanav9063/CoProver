#!/usr/bin/env python3
"""
Parse SLIME training logs and extract metrics over time.

SLIME logs training metrics in lines like:
    step N: {'train/loss': ..., 'train/pg_loss': ..., 'train/entropy_loss': ...,
             'train/grad_norm': ..., 'train/pg_clipfrac': ..., 'train/ppo_kl': ...,
             'train/step': N}

and rollout metrics in lines like:
    {'rollout/raw_reward': ..., 'rollout/response_lengths': ...,
     'rollout/log_probs': ..., 'rollout/step': N}

This script parses both, plots training curves, and computes summary statistics.

Usage:
    python analyze_training.py \
        --log-file /path/to/slime_output.log \
        --output-dir ./training_plots \
        --last-n 50
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("analyze_training")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Pattern for train-step log lines from SLIME (model.py:660):
#   logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")
# role_tag is "" for actor, or e.g. "critic-" for critic
TRAIN_STEP_RE = re.compile(
    r'(?:^|]\s+\S+:\d+\s+-\s+)'      # optional logging prefix
    r'(?P<role>\S*?)step\s+(?P<step>\d+):\s*'
    r'(?P<dict>\{.+\})\s*$'
)

# Pattern for rollout log lines (gather_log_data outputs via wandb/tensorboard)
# These may appear as JSON-like dicts or in wandb log format
ROLLOUT_DICT_RE = re.compile(
    r'(?P<dict>\{[^{}]*["\']rollout/[^{}]+\})'
)


def _safe_parse_dict(s: str) -> Optional[Dict[str, Any]]:
    """Parse a Python dict literal string, handling common edge cases."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    # Try cleaning up common issues (tensor values, inf/nan)
    cleaned = s
    cleaned = re.sub(r'tensor\(([^)]+)\)', r'\1', cleaned)
    cleaned = re.sub(r"inf\b", "'inf'", cleaned)
    cleaned = re.sub(r"nan\b", "'nan'", cleaned)
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    # Last resort: try JSON
    try:
        return json.loads(s.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        pass

    return None


@dataclass
class TrainingMetrics:
    """Container for parsed training metrics."""

    # Train metrics keyed by step
    train_steps: List[int] = field(default_factory=list)
    train_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Rollout metrics keyed by rollout step
    rollout_steps: List[int] = field(default_factory=list)
    rollout_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def parse_log_file(log_path: str) -> TrainingMetrics:
    """Parse a SLIME training log file and extract metrics.

    Args:
        log_path: Path to the log file (plain text).

    Returns:
        TrainingMetrics with parsed train and rollout metrics.
    """
    metrics = TrainingMetrics()
    train_lines_parsed = 0
    rollout_lines_parsed = 0
    total_lines = 0

    with open(log_path) as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            # Try to match a training step line
            m = TRAIN_STEP_RE.search(line)
            if m:
                step = int(m.group("step"))
                dict_str = m.group("dict")
                parsed = _safe_parse_dict(dict_str)
                if parsed is not None:
                    metrics.train_steps.append(step)
                    for key, val in parsed.items():
                        if isinstance(val, (int, float)):
                            metrics.train_metrics[key].append(float(val))
                        elif isinstance(val, str) and val in ("inf", "nan"):
                            metrics.train_metrics[key].append(float(val))
                    train_lines_parsed += 1
                continue

            # Try to match rollout dict lines
            m = ROLLOUT_DICT_RE.search(line)
            if m:
                dict_str = m.group("dict")
                parsed = _safe_parse_dict(dict_str)
                if parsed is not None:
                    step = parsed.get("rollout/step", len(metrics.rollout_steps))
                    if isinstance(step, (int, float)):
                        step = int(step)
                    metrics.rollout_steps.append(step)
                    for key, val in parsed.items():
                        if isinstance(val, (int, float)):
                            metrics.rollout_metrics[key].append(float(val))
                    rollout_lines_parsed += 1

    logger.info(
        "Parsed %d lines: %d train steps, %d rollout steps",
        total_lines, train_lines_parsed, rollout_lines_parsed,
    )

    if train_lines_parsed == 0 and rollout_lines_parsed == 0:
        logger.warning("No metrics found in log file. Check the file format.")

    return metrics


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(
    metrics: TrainingMetrics,
    last_n: int = 50,
) -> Dict[str, Any]:
    """Compute summary statistics over the training run.

    Args:
        metrics: Parsed training metrics.
        last_n: Number of recent steps to use for "recent" statistics.

    Returns:
        Dict of summary statistics.
    """
    summary = {
        "total_train_steps": len(metrics.train_steps),
        "total_rollout_steps": len(metrics.rollout_steps),
    }

    # Key metrics to summarize
    key_train_metrics = [
        "train/loss", "train/pg_loss", "train/entropy_loss",
        "train/grad_norm", "train/pg_clipfrac", "train/ppo_kl",
        "train/kl_loss",
    ]

    key_rollout_metrics = [
        "rollout/raw_reward", "rollout/response_lengths",
        "rollout/log_probs", "rollout/ref_log_probs",
        "rollout/entropy",
    ]

    for key in key_train_metrics:
        vals = metrics.train_metrics.get(key, [])
        if not vals:
            continue

        arr = np.array(vals, dtype=np.float64)
        # Filter out inf/nan for statistics
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            continue

        recent = valid[-last_n:] if len(valid) >= last_n else valid

        summary[key] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "final": float(valid[-1]),
            f"mean_last_{last_n}": float(np.mean(recent)),
            f"std_last_{last_n}": float(np.std(recent)),
        }

        # Compute reward trend (linear regression slope over last N steps)
        if len(recent) >= 5:
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            summary[key]["trend_slope"] = float(coeffs[0])
            summary[key]["trend_direction"] = "improving" if (
                (key == "train/loss" and coeffs[0] < 0) or
                (key != "train/loss" and coeffs[0] > 0)
            ) else "degrading"

    for key in key_rollout_metrics:
        vals = metrics.rollout_metrics.get(key, [])
        if not vals:
            continue

        arr = np.array(vals, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            continue

        recent = valid[-last_n:] if len(valid) >= last_n else valid

        summary[key] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "final": float(valid[-1]),
            f"mean_last_{last_n}": float(np.mean(recent)),
        }

        if len(recent) >= 5:
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            summary[key]["trend_slope"] = float(coeffs[0])

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    metrics: TrainingMetrics,
    output_dir: str,
    smoothing_window: int = 10,
) -> List[str]:
    """Plot training curves and save to output directory.

    Args:
        metrics: Parsed training metrics.
        output_dir: Directory to save plot images.
        smoothing_window: Window size for exponential moving average smoothing.

    Returns:
        List of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with: pip install matplotlib")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        """Exponential moving average smoothing."""
        if len(values) <= 1 or window <= 1:
            return values
        alpha = 2.0 / (window + 1)
        result = np.empty_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    def _plot_metric(
        steps: List[int],
        values: List[float],
        title: str,
        ylabel: str,
        filename: str,
        smooth: bool = True,
    ) -> Optional[str]:
        if not values:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))

        arr = np.array(values, dtype=np.float64)
        x = np.array(steps[:len(arr)])

        # Raw data (transparent)
        ax.plot(x, arr, alpha=0.25, color="tab:blue", linewidth=0.8, label="raw")

        # Smoothed
        if smooth and len(arr) > smoothing_window:
            smoothed = _smooth(arr, smoothing_window)
            ax.plot(x, smoothed, color="tab:blue", linewidth=2.0,
                    label=f"EMA (window={smoothing_window})")

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", filepath)
        return filepath

    # ---- Train metrics ----

    # 1. Combined loss plot (pg_loss, entropy_loss, kl_loss on same axes)
    loss_keys = ["train/pg_loss", "train/entropy_loss", "train/kl_loss"]
    available_losses = {k: metrics.train_metrics[k] for k in loss_keys if k in metrics.train_metrics}
    if available_losses:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {"train/pg_loss": "tab:blue", "train/entropy_loss": "tab:orange", "train/kl_loss": "tab:green"}
        for key, vals in available_losses.items():
            arr = np.array(vals, dtype=np.float64)
            x = np.array(metrics.train_steps[:len(arr)])
            label = key.replace("train/", "")
            ax.plot(x, arr, alpha=0.2, color=colors.get(key, "gray"), linewidth=0.8)
            if len(arr) > smoothing_window:
                smoothed = _smooth(arr, smoothing_window)
                ax.plot(x, smoothed, color=colors.get(key, "gray"), linewidth=2.0, label=label)
            else:
                ax.plot(x, arr, color=colors.get(key, "gray"), linewidth=1.5, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        filepath = os.path.join(output_dir, "losses.png")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved_files.append(filepath)
        logger.info("Saved plot: %s", filepath)

    # 2. Total loss
    path = _plot_metric(
        metrics.train_steps,
        metrics.train_metrics.get("train/loss", []),
        "Total Training Loss", "Loss", "total_loss.png",
    )
    if path:
        saved_files.append(path)

    # 3. Gradient norm
    path = _plot_metric(
        metrics.train_steps,
        metrics.train_metrics.get("train/grad_norm", []),
        "Gradient Norm", "Grad Norm", "grad_norm.png",
    )
    if path:
        saved_files.append(path)

    # 4. PPO KL divergence
    path = _plot_metric(
        metrics.train_steps,
        metrics.train_metrics.get("train/ppo_kl", []),
        "PPO KL Divergence", "KL", "ppo_kl.png",
    )
    if path:
        saved_files.append(path)

    # 5. Policy gradient clip fraction
    path = _plot_metric(
        metrics.train_steps,
        metrics.train_metrics.get("train/pg_clipfrac", []),
        "PG Clip Fraction", "Clip Fraction", "pg_clipfrac.png",
    )
    if path:
        saved_files.append(path)

    # ---- Rollout metrics ----

    # 6. Raw reward
    path = _plot_metric(
        metrics.rollout_steps,
        metrics.rollout_metrics.get("rollout/raw_reward", []),
        "Raw Reward (per rollout)", "Reward", "raw_reward.png",
    )
    if path:
        saved_files.append(path)

    # 7. Response lengths
    path = _plot_metric(
        metrics.rollout_steps,
        metrics.rollout_metrics.get("rollout/response_lengths", []),
        "Response Lengths (per rollout)", "Tokens", "response_lengths.png",
    )
    if path:
        saved_files.append(path)

    # 8. Rollout entropy
    path = _plot_metric(
        metrics.rollout_steps,
        metrics.rollout_metrics.get("rollout/entropy", []),
        "Rollout Entropy", "Entropy", "rollout_entropy.png",
    )
    if path:
        saved_files.append(path)

    # 9. Combined overview (2x2 subplot)
    overview_metrics = [
        ("train/loss", "Total Loss"),
        ("train/grad_norm", "Gradient Norm"),
    ]
    overview_rollout = [
        ("rollout/raw_reward", "Raw Reward"),
    ]

    # Collect what we have
    panel_data = []
    for key, label in overview_metrics:
        vals = metrics.train_metrics.get(key, [])
        steps = metrics.train_steps[:len(vals)]
        if vals:
            panel_data.append((steps, vals, label))
    for key, label in overview_rollout:
        vals = metrics.rollout_metrics.get(key, [])
        steps = metrics.rollout_steps[:len(vals)]
        if vals:
            panel_data.append((steps, vals, label))

    if len(panel_data) >= 2:
        n_panels = min(len(panel_data), 4)
        ncols = 2
        nrows = (n_panels + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        for idx in range(n_panels):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            steps, vals, label = panel_data[idx]
            arr = np.array(vals, dtype=np.float64)
            x = np.array(steps)
            ax.plot(x, arr, alpha=0.25, color="tab:blue", linewidth=0.8)
            if len(arr) > smoothing_window:
                ax.plot(x, _smooth(arr, smoothing_window), color="tab:blue", linewidth=2.0)
            ax.set_title(label)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        # Hide unused panels
        for idx in range(n_panels, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        filepath = os.path.join(output_dir, "overview.png")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved_files.append(filepath)
        logger.info("Saved plot: %s", filepath)

    # Plot any additional train metrics not covered above
    covered_train = {
        "train/loss", "train/pg_loss", "train/entropy_loss", "train/kl_loss",
        "train/grad_norm", "train/ppo_kl", "train/pg_clipfrac", "train/step",
    }
    for key, vals in sorted(metrics.train_metrics.items()):
        if key in covered_train or not vals:
            continue
        safe_name = key.replace("/", "_").replace(" ", "_")
        path = _plot_metric(
            metrics.train_steps[:len(vals)], vals,
            key, key.split("/")[-1], f"{safe_name}.png",
        )
        if path:
            saved_files.append(path)

    logger.info("Generated %d plots in %s", len(saved_files), output_dir)
    return saved_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SLIME training logs and produce training curves and statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-file", required=True,
        help="Path to the SLIME task output log file.",
    )
    parser.add_argument(
        "--output-dir", default="./training_plots",
        help="Directory to save plot images.",
    )
    parser.add_argument(
        "--last-n", type=int, default=50,
        help="Number of recent steps for summary statistics.",
    )
    parser.add_argument(
        "--smoothing-window", type=int, default=10,
        help="EMA smoothing window for plots.",
    )
    parser.add_argument(
        "--summary-output", type=str, default=None,
        help="Path to save summary statistics as JSON.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (only compute summary stats).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        logger.error("Log file not found: %s", args.log_file)
        sys.exit(1)

    logger.info("Parsing log file: %s", args.log_file)
    metrics = parse_log_file(args.log_file)

    # Print discovered metric keys
    if metrics.train_metrics:
        logger.info("Train metrics found: %s", sorted(metrics.train_metrics.keys()))
    if metrics.rollout_metrics:
        logger.info("Rollout metrics found: %s", sorted(metrics.rollout_metrics.keys()))

    # Compute summary
    summary = compute_summary(metrics, last_n=args.last_n)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING SUMMARY (last_n=%d)", args.last_n)
    logger.info("=" * 70)
    logger.info("Total train steps: %d", summary["total_train_steps"])
    logger.info("Total rollout steps: %d", summary["total_rollout_steps"])

    for key in sorted(summary.keys()):
        if key.startswith(("total_", )):
            continue
        val = summary[key]
        if isinstance(val, dict):
            logger.info("")
            logger.info("  %s:", key)
            for subkey, subval in val.items():
                if isinstance(subval, float):
                    logger.info("    %-25s %.6f", subkey, subval)
                else:
                    logger.info("    %-25s %s", subkey, subval)

    logger.info("=" * 70)

    # Save summary JSON
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = os.path.join(args.output_dir, "summary.json")

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Generate plots
    if not args.no_plots:
        saved_files = plot_training_curves(
            metrics, args.output_dir,
            smoothing_window=args.smoothing_window,
        )
        if not saved_files:
            logger.warning("No plots were generated. Check if the log file contains parseable metrics.")
    else:
        logger.info("Skipping plot generation (--no-plots).")


if __name__ == "__main__":
    main()
