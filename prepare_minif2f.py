"""
Download and prepare MiniF2F and Kimina-Prover-Promptset datasets from HuggingFace.

Outputs:
  - data/minif2f_test.jsonl   (AI-MO/minif2f_test)
  - data/kimina_train.jsonl   (AI-MO/Kimina-Prover-Promptset, train split)
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}\n```\n"
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def prepare_minif2f(output_dir: str) -> str:
    """Download AI-MO/minif2f_test and save as JSONL."""
    print("Downloading AI-MO/minif2f_test ...")
    ds = load_dataset("AI-MO/minif2f_test", split="test")

    out_path = os.path.join(output_dir, "minif2f_test.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in ds:
            formal_statement = row.get("formal_statement", row.get("statement", ""))
            name = row.get("name", row.get("id", f"minif2f_{count}"))
            record = {
                "prompt": PROMPT_TEMPLATE.format(formal_statement=formal_statement),
                "formal_statement": formal_statement,
                "name": str(name),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Saved {count} examples to {out_path}")
    return out_path


def prepare_kimina(output_dir: str) -> str:
    """Download AI-MO/Kimina-Prover-Promptset (train split) and save as JSONL."""
    print("Downloading AI-MO/Kimina-Prover-Promptset ...")
    ds = load_dataset("AI-MO/Kimina-Prover-Promptset", split="train")

    out_path = os.path.join(output_dir, "kimina_train.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in ds:
            formal_statement = row.get("formal_statement", row.get("statement", ""))
            name = row.get("name", row.get("id", f"kimina_{count}"))
            record = {
                "prompt": PROMPT_TEMPLATE.format(formal_statement=formal_statement),
                "formal_statement": formal_statement,
                "name": str(name),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Saved {count} examples to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare MiniF2F and Kimina-Prover-Promptset datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--minif2f-only",
        action="store_true",
        help="Only download MiniF2F test set",
    )
    parser.add_argument(
        "--kimina-only",
        action="store_true",
        help="Only download Kimina-Prover-Promptset",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.kimina_only:
        prepare_kimina(args.output_dir)
    elif args.minif2f_only:
        prepare_minif2f(args.output_dir)
    else:
        prepare_minif2f(args.output_dir)
        prepare_kimina(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
