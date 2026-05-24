#!/usr/bin/env python3
"""
Whole-proof evaluation: generate proofs with SGLang, verify with kimina-lean-server.

Supports pass@1, pass@8, pass@32 with concurrent verification.

Usage:
    python evaluate.py \
        --dataset data/minif2f.jsonl \
        --sglang-url http://localhost:30000 \
        --kimina-url http://localhost:8000 \
        --n-samples 32 \
        --output results/eval_results.json
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
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# pass@k estimator (unbiased, from Chen et al. "Evaluating Large Language
# Models Trained on Code")
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k.

    Args:
        n: total number of samples generated per problem.
        c: number of correct samples for a problem.
        k: k in pass@k.

    Returns:
        Estimated probability of at least one correct sample in k draws.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Proof generation (synchronous, batched via SGLang /v1/completions)
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
    if n == 1:
        payload["temperature"] = 0.0

    resp = requests.post(
        f"{sglang_url}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return [c["text"] for c in data["choices"]]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_lean_code(response: str) -> Optional[str]:
    """Extract the last lean4 code block from model output.

    Falls back to treating the entire response as code if no block found
    and the response looks like Lean code.
    """
    blocks = re.findall(r'```lean4?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        code = blocks[-1].strip()
    else:
        # Heuristic: if the response contains theorem/def/lemma, treat as raw code
        if any(kw in response for kw in ["theorem ", "def ", "lemma ", "example ", "sorry"]):
            code = response.strip()
        else:
            return None

    if not code:
        return None

    # Ensure Mathlib import is present
    if "import Mathlib" not in code and "import " not in code:
        code = "import Mathlib\n\n" + code

    return code


# ---------------------------------------------------------------------------
# Async verification via kimina-lean-server
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
        # kimina-lean-server returns empty response field when proof is valid
        response_field = r.get("response")
        if not response_field:
            return True
        # If response is a dict, check for error messages
        if isinstance(response_field, dict):
            return not response_field.get("messages")
        # If response is a non-empty string, proof failed
        return not bool(response_field)

    except asyncio.TimeoutError:
        logger.debug("Verification timed out for problem %s", problem_id)
        return False
    except Exception as e:
        logger.debug("Verification error for problem %s: %s", problem_id, e)
        return False


async def verify_batch_async(
    codes: List[Optional[str]],
    kimina_url: str,
    proof_timeout: int = 120,
    max_concurrent: int = 16,
    problem_id: str = "0",
) -> List[bool]:
    """Verify multiple proofs concurrently with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_verify(session, code, idx):
        async with semaphore:
            return await verify_proof_async(
                session, code, kimina_url, proof_timeout,
                problem_id=f"{problem_id}_{idx}",
            )

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _limited_verify(session, code, i)
            for i, code in enumerate(codes)
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate_benchmark(
    dataset_path: str,
    sglang_url: str,
    kimina_url: str,
    n_samples: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    max_problems: Optional[int] = None,
    max_concurrent_verify: int = 16,
    proof_timeout: int = 120,
) -> Tuple[Dict[str, Any], Dict[int, float]]:
    """Evaluate on a benchmark dataset.

    Returns:
        Tuple of (per-problem results dict, pass@k rates dict).
    """
    # Load dataset
    with open(dataset_path) as f:
        if dataset_path.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if max_problems is not None and max_problems > 0:
        data = data[:max_problems]

    total_problems = len(data)
    logger.info("Loaded %d problems from %s", total_problems, dataset_path)
    logger.info(
        "Configuration: n_samples=%d, temperature=%.2f, max_tokens=%d",
        n_samples, temperature, max_tokens,
    )

    # Determine which k values to report
    k_values = sorted(set(k for k in [1, 8, 32] if k <= n_samples))
    logger.info("Will report pass@k for k=%s", k_values)

    results = {}
    pass_k_correct = defaultdict(int)  # k -> count of problems with >= 1 correct in k
    all_correct_counts = []  # (n, c) pairs for unbiased pass@k

    start_time = time.time()

    for i, sample in enumerate(data):
        # Extract the formal statement
        if "formal_statement" in sample:
            formal = sample["formal_statement"]
        elif "prompt" in sample:
            formal = sample["prompt"]
        elif "statement" in sample:
            formal = sample["statement"]
        else:
            logger.warning("Problem %d: no formal_statement/prompt/statement field, skipping", i)
            continue

        name = sample.get("name", sample.get("statement_id", sample.get("full_name", str(i))))

        # Build prompt for the model
        prompt = f"Complete the following Lean 4 code:\n\n```lean4\n{formal}\n```\n"

        # Generate proof attempts
        try:
            outputs = generate_proofs(
                prompt, sglang_url,
                n=n_samples, temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("[%d/%d] %s: generation failed: %s", i + 1, total_problems, name, e)
            results[name] = {
                "proved": False,
                "num_proved": 0,
                "num_attempts": n_samples,
                "error": f"generation_failed: {e}",
            }
            all_correct_counts.append((n_samples, 0))
            continue

        # Extract code from each output
        codes = [extract_lean_code(o) for o in outputs]
        n_extracted = sum(1 for c in codes if c is not None)

        # Verify all proofs concurrently
        try:
            verified = await verify_batch_async(
                codes, kimina_url,
                proof_timeout=proof_timeout,
                max_concurrent=max_concurrent_verify,
                problem_id=name,
            )
        except Exception as e:
            logger.error("[%d/%d] %s: verification failed: %s", i + 1, total_problems, name, e)
            verified = [False] * len(codes)

        num_proved = sum(verified)
        any_proved = num_proved > 0

        results[name] = {
            "proved": any_proved,
            "num_proved": num_proved,
            "num_attempts": n_samples,
            "num_extracted": n_extracted,
        }
        all_correct_counts.append((n_samples, num_proved))

        # Track empirical pass@k (for each k, check if any of the first k are correct)
        for k in k_values:
            if any(verified[:k]):
                pass_k_correct[k] += 1

        status = "PROVED" if any_proved else "failed"
        logger.info(
            "[%d/%d] %s: %s (%d/%d correct, %d extracted)",
            i + 1, total_problems, name, status, num_proved, n_samples, n_extracted,
        )

        # Progress report every 10 problems
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_problems - i - 1) / rate if rate > 0 else 0
            logger.info("")
            logger.info("--- Progress: %d/%d (%.1f/s, ETA %.0fs) ---", i + 1, total_problems, rate, eta)
            for k in k_values:
                empirical = pass_k_correct[k] / (i + 1)
                logger.info("  pass@%d (empirical): %d/%d = %.4f", k, pass_k_correct[k], i + 1, empirical)
            logger.info("")

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    elapsed = time.time() - start_time
    total_evaluated = len(results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS (%d problems, %.1f seconds)", total_evaluated, elapsed)
    logger.info("=" * 70)

    pass_at_k_rates = {}

    for k in k_values:
        # Unbiased pass@k estimator
        if total_evaluated > 0:
            unbiased = sum(
                pass_at_k(n, c, k) for n, c in all_correct_counts
            ) / total_evaluated
        else:
            unbiased = 0.0

        # Empirical pass@k (simple: did any of first k attempts succeed?)
        empirical = pass_k_correct[k] / max(total_evaluated, 1)

        pass_at_k_rates[k] = {
            "unbiased": unbiased,
            "empirical": empirical,
            "count": pass_k_correct[k],
        }

        logger.info(
            "  pass@%d: %.4f (unbiased) | %.4f (empirical, %d/%d)",
            k, unbiased, empirical, pass_k_correct[k], total_evaluated,
        )

    total_proved = sum(1 for r in results.values() if r["proved"])
    logger.info("  Total proved (any@%d): %d/%d = %.4f", n_samples, total_proved, total_evaluated,
                total_proved / max(total_evaluated, 1))
    logger.info("=" * 70)

    return results, pass_at_k_rates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whole-proof evaluation: generate with SGLang, verify with kimina-lean-server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", required=True,
        help="Path to benchmark dataset (JSON or JSONL with formal_statement field).",
    )
    parser.add_argument(
        "--sglang-url", default="http://localhost:30000",
        help="SGLang server base URL.",
    )
    parser.add_argument(
        "--kimina-url", default="http://localhost:8000",
        help="kimina-lean-server base URL.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=8,
        help="Number of proof attempts per problem (controls pass@k upper bound).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (0.0 for greedy when n_samples=1).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Maximum tokens per generated proof.",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit number of problems for quick testing.",
    )
    parser.add_argument(
        "--max-concurrent-verify", type=int, default=16,
        help="Maximum concurrent verification requests to kimina-lean-server.",
    )
    parser.add_argument(
        "--proof-timeout", type=int, default=120,
        help="Timeout in seconds for each proof verification.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    results, pass_at_k_rates = asyncio.run(
        evaluate_benchmark(
            dataset_path=args.dataset,
            sglang_url=args.sglang_url,
            kimina_url=args.kimina_url,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_problems=args.max_problems,
            max_concurrent_verify=args.max_concurrent_verify,
            proof_timeout=args.proof_timeout,
        )
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "config": {
                "dataset": args.dataset,
                "n_samples": args.n_samples,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "sglang_url": args.sglang_url,
                "kimina_url": args.kimina_url,
            },
            "pass_at_k": {
                str(k): v for k, v in pass_at_k_rates.items()
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
