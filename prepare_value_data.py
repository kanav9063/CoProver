"""
Convert trajectory JSONL (from trajectory_collector.py) to value model SFT format
compatible with SLIME's SFT loss mode.

Input format  (one JSON object per line):
  {"state": "...", "label": 0.77, "theorem": "...", "proved": true/false, ...}

Labels are gamma^d values where d = remaining steps to QED, gamma = 0.95.
  - Proved states: 0.0 < label <= 1.0 (closer to 1.0 = fewer steps remaining)
  - Failed states: label = 0.0

Output format (one JSON object per line):
  {"prompt": "Estimate the discounted distance...\n{state}\nValue: ",
   "label": "0.77"}

Supports oversampling of positive (proved=true) examples to handle class imbalance.
"""

import argparse
import json
import os
import random
from pathlib import Path


VALUE_PROMPT_TEMPLATE = (
    "Estimate the discounted distance to proof completion for the following "
    "Lean 4 proof state. Output a value between 0.0 (dead end or far from done) "
    "and 1.0 (very close to QED).\n\nProof state:\n{state}\n\nValue: "
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_trajectories(input_path: str) -> list[dict]:
    """Load trajectory JSONL file."""
    entries = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_num}: {e}")
    return entries


def convert_entry(entry: dict) -> dict:
    """Convert a single trajectory entry to SFT format."""
    state = entry.get("state", "")
    label_val = float(entry.get("label", 0.0))
    # Format label as two-decimal string
    label_str = f"{label_val:.2f}"

    # If the state already contains the prompt template (trajectory_collector
    # pre-formats it), use it directly; otherwise wrap it.
    if "Rate the following Lean 4 proof state" in state:
        prompt = state
    else:
        prompt = VALUE_PROMPT_TEMPLATE.format(state=state)

    return {"prompt": prompt, "label": label_str}


def oversample(
    positives: list[dict],
    negatives: list[dict],
    factor: float,
) -> list[dict]:
    """Oversample positives by the given factor and merge with negatives.

    factor=1.0 means no oversampling (keep as-is).
    factor=3.0 means repeat each positive example ~3 times.
    Non-integer factors are handled by repeating floor(factor) times and then
    randomly sampling the remainder.
    """
    if factor <= 0:
        raise ValueError("Oversample factor must be > 0")

    full_repeats = int(factor)
    fractional = factor - full_repeats

    oversampled = list(positives) * full_repeats
    if fractional > 0:
        extra_count = int(len(positives) * fractional)
        if extra_count > 0:
            oversampled.extend(random.sample(positives, min(extra_count, len(positives))))

    combined = oversampled + negatives
    random.shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory JSONL to SLIME value-model SFT format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input trajectory JSONL file (from trajectory_collector.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: data/value_sft.jsonl)",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.0,
        help="Oversample positive (proved=true) examples by this factor (default: 1.0 = no oversampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, "value_sft.jsonl")

    random.seed(args.seed)

    # Load raw trajectories
    raw = load_trajectories(args.input)
    print(f"Loaded {len(raw)} trajectory entries from {args.input}")

    # Split into positive/negative and convert
    positives = []
    negatives = []
    for entry in raw:
        converted = convert_entry(entry)
        proved = entry.get("proved", False)
        if proved or float(entry.get("label", 0.0)) > 0.5:
            positives.append(converted)
        else:
            negatives.append(converted)

    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")

    if args.oversample_factor != 1.0:
        print(f"  Oversampling positives by factor {args.oversample_factor}")
        combined = oversample(positives, negatives, args.oversample_factor)
    else:
        combined = positives + negatives
        random.shuffle(combined)

    print(f"  Total after oversampling: {len(combined)}")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for record in combined:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(combined)} SFT examples to {args.output}")


if __name__ == "__main__":
    main()
