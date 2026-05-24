"""
Dataset difficulty filtering.

Run the model on the full dataset with pass@8, then filter to keep only
"frontier" problems where 1 <= num_proved <= 6 out of 8 attempts.

Problems that are always solved (8/8) are too easy.
Problems that are never solved (0/8) are too hard.
Frontier problems have the most GRPO signal.
"""

import json
import re
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_and_verify(prompt, sglang_url, kimina_url, n=8, temperature=0.8, max_tokens=4096):
    """Generate n proofs and verify each. Return number that passed."""
    try:
        resp = requests.post(f"{sglang_url}/v1/completions", json={
            "model": "default",
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, timeout=300)
        choices = resp.json()["choices"]
    except Exception:
        return 0

    passed = 0
    for choice in choices:
        output = choice["text"]
        blocks = re.findall(r'```lean4?\s*\n(.*?)```', output, re.DOTALL)
        if not blocks:
            continue
        code = blocks[-1].strip()
        if "import Mathlib" not in code:
            code = "import Mathlib\n\n" + code
        try:
            verify = requests.post(f"{kimina_url}/api/check", json={
                "snippets": [{"id": "0", "code": code}],
                "timeout": 60,
            }, timeout=90)
            result = verify.json()["results"][0]
            if not bool(result.get("response")):
                passed += 1
        except Exception:
            continue

    return passed


def filter_dataset(args):
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    if args.max_problems:
        data = data[:args.max_problems]

    print(f"Filtering {len(data)} problems with pass@{args.n_samples}")

    easy = []    # always solved
    frontier = [] # sometimes solved (the sweet spot)
    hard = []    # never solved

    for i, sample in enumerate(data):
        prompt = sample.get("prompt", "")
        if not prompt:
            continue

        num_passed = generate_and_verify(
            prompt, args.sglang_url, args.kimina_url,
            n=args.n_samples, temperature=args.temperature,
        )

        if num_passed == 0:
            hard.append(sample)
        elif num_passed == args.n_samples:
            easy.append(sample)
        else:
            frontier.append(sample)
            sample["_pass_rate"] = num_passed / args.n_samples

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(data)}] easy={len(easy)} frontier={len(frontier)} hard={len(hard)}")

    print(f"\nFinal: easy={len(easy)} frontier={len(frontier)} hard={len(hard)}")
    print(f"Frontier pass rate: {sum(s['_pass_rate'] for s in frontier) / max(len(frontier), 1):.2f}")

    # Save frontier dataset
    with open(args.output, "w") as f:
        for s in frontier:
            f.write(json.dumps(s) + "\n")
    print(f"Frontier dataset saved to {args.output} ({len(frontier)} problems)")

    # Also save easy for potential curriculum
    if args.save_easy:
        with open(args.save_easy, "w") as f:
            for s in easy:
                f.write(json.dumps(s) + "\n")
        print(f"Easy dataset saved to {args.save_easy} ({len(easy)} problems)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--kimina-url", default="http://localhost:8000")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--save-easy", type=str, default=None)
    args = parser.parse_args()

    filter_dataset(args)
