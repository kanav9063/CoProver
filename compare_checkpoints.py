#!/usr/bin/env python3
"""
Compare two model checkpoints on a benchmark by running the same problems
through two SGLang instances and verifying with kimina-lean-server.

Reports which model proves more theorems, and identifies problems each model
solves uniquely. Useful for before/after GRPO comparison.

Usage:
    python compare_checkpoints.py \
        --dataset data/minif2f.jsonl \
        --sglang-url-a http://localhost:30000 \
        --sglang-url-b http://localhost:30001 \
        --label-a "base-model" \
        --label-b "grpo-round-3" \
        --kimina-url http://localhost:8000 \
        --n-samples 8 \
        --output comparison_results.json
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("compare_checkpoints")


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Proof generation
# ---------------------------------------------------------------------------

def generate_proofs(
    prompt: str,
    sglang_url: str,
    n: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> List[str]:
    """Generate n proof attempts for a single prompt via SGLang."""
    import requests

    payload = {
        "model": "default",
        "prompt": prompt,
        "n": n,
        "temperature": temperature if n > 1 else 0.0,
        "max_tokens": max_tokens,
    }

    resp = requests.post(
        f"{sglang_url}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return [c["text"] for c in resp.json()["choices"]]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_lean_code(response: str) -> Optional[str]:
    """Extract the last lean4 code block from model output."""
    blocks = re.findall(r'```lean4?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        code = blocks[-1].strip()
    else:
        if any(kw in response for kw in ["theorem ", "def ", "lemma ", "example ", "sorry"]):
            code = response.strip()
        else:
            return None

    if not code:
        return None

    if "import Mathlib" not in code and "import " not in code:
        code = "import Mathlib\n\n" + code

    return code


# ---------------------------------------------------------------------------
# Async verification
# ---------------------------------------------------------------------------

async def verify_proof_async(
    session: aiohttp.ClientSession,
    code: Optional[str],
    kimina_url: str,
    proof_timeout: int = 120,
    problem_id: str = "0",
) -> bool:
    """Verify a single proof asynchronously with kimina-lean-server."""
    if not code:
        return False

    try:
        payload = {
            "snippets": [{"id": problem_id, "code": code}],
            "timeout": proof_timeout,
        }
        async with session.post(
            f"{kimina_url}/api/check",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=proof_timeout + 60),
        ) as resp:
            data = await resp.json()

        results = data.get("results", [])
        if not results:
            return False

        r = results[0]
        response_field = r.get("response")
        if not response_field:
            return True
        if isinstance(response_field, dict):
            return not response_field.get("messages")
        return not bool(response_field)

    except (asyncio.TimeoutError, Exception):
        return False


async def verify_batch_async(
    codes: List[Optional[str]],
    kimina_url: str,
    proof_timeout: int = 120,
    max_concurrent: int = 16,
    problem_id: str = "0",
) -> List[bool]:
    """Verify multiple proofs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_verify(session, code, idx):
        async with semaphore:
            return await verify_proof_async(
                session, code, kimina_url, proof_timeout,
                problem_id=f"{problem_id}_{idx}",
            )

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_limited_verify(session, code, i) for i, code in enumerate(codes)]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Single-model evaluation (returns per-problem results)
# ---------------------------------------------------------------------------

async def evaluate_model(
    data: List[Dict],
    sglang_url: str,
    kimina_url: str,
    label: str,
    n_samples: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    max_concurrent_verify: int = 16,
    proof_timeout: int = 120,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a single model on the given problems.

    Returns:
        Dict mapping problem name -> {proved, num_proved, num_attempts}.
    """
    results = {}
    total = len(data)

    logger.info("[%s] Starting evaluation on %d problems (n=%d)", label, total, n_samples)

    for i, sample in enumerate(data):
        if "formal_statement" in sample:
            formal = sample["formal_statement"]
        elif "prompt" in sample:
            formal = sample["prompt"]
        elif "statement" in sample:
            formal = sample["statement"]
        else:
            continue

        name = sample.get("name", sample.get("statement_id", sample.get("full_name", str(i))))
        prompt = f"Complete the following Lean 4 code:\n\n```lean4\n{formal}\n```\n"

        # Generate
        try:
            outputs = generate_proofs(
                prompt, sglang_url,
                n=n_samples, temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("[%s] [%d/%d] %s: generation failed: %s", label, i + 1, total, name, e)
            results[name] = {"proved": False, "num_proved": 0, "num_attempts": n_samples}
            continue

        # Extract and verify
        codes = [extract_lean_code(o) for o in outputs]
        try:
            verified = await verify_batch_async(
                codes, kimina_url,
                proof_timeout=proof_timeout,
                max_concurrent=max_concurrent_verify,
                problem_id=f"{label}_{name}",
            )
        except Exception:
            verified = [False] * len(codes)

        num_proved = sum(verified)
        results[name] = {
            "proved": num_proved > 0,
            "num_proved": num_proved,
            "num_attempts": n_samples,
        }

        if num_proved > 0:
            logger.info("[%s] [%d/%d] PROVED: %s (%d/%d)", label, i + 1, total, name, num_proved, n_samples)

        if (i + 1) % 10 == 0:
            proved_so_far = sum(1 for r in results.values() if r["proved"])
            logger.info("[%s] Progress: %d/%d problems, %d proved so far", label, i + 1, total, proved_so_far)

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(
    results_a: Dict[str, Dict[str, Any]],
    results_b: Dict[str, Dict[str, Any]],
    label_a: str,
    label_b: str,
    n_samples: int,
) -> Dict[str, Any]:
    """Compare results from two models.

    Returns:
        Comparison report dict.
    """
    all_names = sorted(set(results_a.keys()) | set(results_b.keys()))

    proved_a: Set[str] = {n for n in all_names if results_a.get(n, {}).get("proved", False)}
    proved_b: Set[str] = {n for n in all_names if results_b.get(n, {}).get("proved", False)}

    both = proved_a & proved_b
    only_a = proved_a - proved_b
    only_b = proved_b - proved_a
    neither = set(all_names) - proved_a - proved_b

    total = len(all_names)

    # Compute pass@k for both models
    k_values = sorted(set(k for k in [1, 8, 32] if k <= n_samples))
    pass_k_a = {}
    pass_k_b = {}
    for k in k_values:
        scores_a = [
            pass_at_k(
                results_a.get(n, {}).get("num_attempts", n_samples),
                results_a.get(n, {}).get("num_proved", 0),
                k,
            )
            for n in all_names
        ]
        scores_b = [
            pass_at_k(
                results_b.get(n, {}).get("num_attempts", n_samples),
                results_b.get(n, {}).get("num_proved", 0),
                k,
            )
            for n in all_names
        ]
        pass_k_a[k] = sum(scores_a) / max(total, 1)
        pass_k_b[k] = sum(scores_b) / max(total, 1)

    report = {
        "total_problems": total,
        label_a: {
            "proved": len(proved_a),
            "rate": len(proved_a) / max(total, 1),
            "pass_at_k": {str(k): v for k, v in pass_k_a.items()},
        },
        label_b: {
            "proved": len(proved_b),
            "rate": len(proved_b) / max(total, 1),
            "pass_at_k": {str(k): v for k, v in pass_k_b.items()},
        },
        "both_proved": len(both),
        f"only_{label_a}": sorted(only_a),
        f"only_{label_b}": sorted(only_b),
        "neither_proved": len(neither),
        "improvement": {
            "absolute": len(proved_b) - len(proved_a),
            "relative_pct": (
                ((len(proved_b) - len(proved_a)) / max(len(proved_a), 1)) * 100
            ),
        },
        "per_problem": {},
    }

    # Per-problem detail
    for name in all_names:
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        report["per_problem"][name] = {
            f"{label_a}_proved": ra.get("proved", False),
            f"{label_a}_num_proved": ra.get("num_proved", 0),
            f"{label_b}_proved": rb.get("proved", False),
            f"{label_b}_num_proved": rb.get("num_proved", 0),
        }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_comparison(
    dataset_path: str,
    sglang_url_a: str,
    sglang_url_b: str,
    kimina_url: str,
    label_a: str,
    label_b: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    max_problems: Optional[int],
    max_concurrent_verify: int,
    proof_timeout: int,
    parallel_models: bool,
) -> Dict[str, Any]:
    """Run the full comparison."""
    # Load dataset
    with open(dataset_path) as f:
        if dataset_path.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if max_problems is not None and max_problems > 0:
        data = data[:max_problems]

    logger.info("Loaded %d problems from %s", len(data), dataset_path)
    logger.info("Model A (%s): %s", label_a, sglang_url_a)
    logger.info("Model B (%s): %s", label_b, sglang_url_b)

    start_time = time.time()

    eval_kwargs = dict(
        kimina_url=kimina_url,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrent_verify=max_concurrent_verify,
        proof_timeout=proof_timeout,
    )

    if parallel_models:
        # Run both models in parallel (requires separate SGLang servers and
        # enough kimina-lean-server capacity)
        logger.info("Running both models in parallel...")
        results_a_task = evaluate_model(
            data, sglang_url_a, label=label_a, **eval_kwargs,
        )
        results_b_task = evaluate_model(
            data, sglang_url_b, label=label_b, **eval_kwargs,
        )
        results_a, results_b = await asyncio.gather(results_a_task, results_b_task)
    else:
        # Sequential: evaluate model A first, then model B
        logger.info("Evaluating model A (%s) first...", label_a)
        results_a = await evaluate_model(
            data, sglang_url_a, label=label_a, **eval_kwargs,
        )
        logger.info("Evaluating model B (%s)...", label_b)
        results_b = await evaluate_model(
            data, sglang_url_b, label=label_b, **eval_kwargs,
        )

    elapsed = time.time() - start_time

    # Compare
    report = compare_results(results_a, results_b, label_a, label_b, n_samples)
    report["elapsed_seconds"] = round(elapsed, 1)
    report["config"] = {
        "dataset": dataset_path,
        "n_samples": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON RESULTS (%d problems, %.1f seconds)", report["total_problems"], elapsed)
    logger.info("=" * 70)
    logger.info("")

    ra = report[label_a]
    rb = report[label_b]
    logger.info("  %-25s  Proved: %3d/%d (%.4f)", label_a, ra["proved"], report["total_problems"], ra["rate"])
    logger.info("  %-25s  Proved: %3d/%d (%.4f)", label_b, rb["proved"], report["total_problems"], rb["rate"])
    logger.info("")

    # pass@k comparison
    k_values = sorted(set(int(k) for k in ra["pass_at_k"].keys()))
    for k in k_values:
        ka = ra["pass_at_k"][str(k)]
        kb = rb["pass_at_k"][str(k)]
        diff = kb - ka
        arrow = "+" if diff > 0 else ""
        logger.info("  pass@%-3d  %s: %.4f  |  %s: %.4f  |  diff: %s%.4f", k, label_a, ka, label_b, kb, arrow, diff)

    logger.info("")
    logger.info("  Both proved:          %d", report["both_proved"])
    logger.info("  Only %s:  %d", label_a, len(report[f"only_{label_a}"]))
    logger.info("  Only %s:  %d", label_b, len(report[f"only_{label_b}"]))
    logger.info("  Neither proved:       %d", report["neither_proved"])
    logger.info("")

    imp = report["improvement"]
    direction = "improvement" if imp["absolute"] > 0 else "regression" if imp["absolute"] < 0 else "no change"
    logger.info(
        "  Net %s: %+d problems (%+.1f%%)",
        direction, imp["absolute"], imp["relative_pct"],
    )

    # List uniquely solved problems
    only_a_list = report[f"only_{label_a}"]
    only_b_list = report[f"only_{label_b}"]

    if only_a_list:
        logger.info("")
        logger.info("  Problems solved ONLY by %s (%d):", label_a, len(only_a_list))
        for name in only_a_list[:20]:
            logger.info("    - %s", name)
        if len(only_a_list) > 20:
            logger.info("    ... and %d more", len(only_a_list) - 20)

    if only_b_list:
        logger.info("")
        logger.info("  Problems solved ONLY by %s (%d):", label_b, len(only_b_list))
        for name in only_b_list[:20]:
            logger.info("    - %s", name)
        if len(only_b_list) > 20:
            logger.info("    ... and %d more", len(only_b_list) - 20)

    logger.info("")
    logger.info("=" * 70)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two model checkpoints on a Lean4 theorem proving benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", required=True,
        help="Path to benchmark dataset (JSON or JSONL with formal_statement field).",
    )

    # Model endpoints
    parser.add_argument(
        "--sglang-url-a", required=True,
        help="SGLang server URL for model A (e.g., http://localhost:30000).",
    )
    parser.add_argument(
        "--sglang-url-b", required=True,
        help="SGLang server URL for model B (e.g., http://localhost:30001).",
    )
    parser.add_argument(
        "--label-a", default="model-a",
        help="Human-readable label for model A.",
    )
    parser.add_argument(
        "--label-b", default="model-b",
        help="Human-readable label for model B.",
    )

    # Verification
    parser.add_argument(
        "--kimina-url", default="http://localhost:8000",
        help="kimina-lean-server base URL.",
    )

    # Sampling
    parser.add_argument(
        "--n-samples", type=int, default=8,
        help="Number of proof attempts per problem per model.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Maximum tokens per generated proof.",
    )

    # Limits
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit number of problems for quick testing.",
    )
    parser.add_argument(
        "--max-concurrent-verify", type=int, default=16,
        help="Maximum concurrent verification requests.",
    )
    parser.add_argument(
        "--proof-timeout", type=int, default=120,
        help="Timeout in seconds for each proof verification.",
    )

    # Execution
    parser.add_argument(
        "--parallel", action="store_true",
        help="Evaluate both models in parallel (requires both SGLang servers running).",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save comparison results as JSON.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    report = asyncio.run(
        run_comparison(
            dataset_path=args.dataset,
            sglang_url_a=args.sglang_url_a,
            sglang_url_b=args.sglang_url_b,
            kimina_url=args.kimina_url,
            label_a=args.label_a,
            label_b=args.label_b,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_problems=args.max_problems,
            max_concurrent_verify=args.max_concurrent_verify,
            proof_timeout=args.proof_timeout,
            parallel_models=args.parallel,
        )
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
