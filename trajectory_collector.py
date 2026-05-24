"""
Trajectory Collector for Value Model Training (Whole-Proof Generation).

Loads theorems from the LeanDojo benchmark, generates complete proof attempts
via SGLang, verifies them with kimina-lean-server, and records trajectories
as JSONL for training the value model.

Each trajectory entry:
  - state:    the theorem statement formatted as a value model prompt
  - label:    1.0 if verified, 0.0 otherwise
  - theorem:  theorem name
  - proved:   boolean (whether this particular attempt verified)
  - response: the model's raw output

Labels are binary: 1.0 for proofs that pass verification, 0.0 for those that
do not. The "state" field is pre-formatted with the value model prompt template
so the output can be used directly as SFT training data.
"""

import json
import re
import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALUE_MODEL_PROMPT_TEMPLATE = (
    "Estimate the discounted distance to proof completion for the following "
    "Lean 4 proof state. Output a value between 0.0 (dead end or far from done) "
    "and 1.0 (very close to QED).\n\nProof state:\n{state}\n\nValue: "
)

# Discount factor for depth-based labels: label = gamma^d
# where d = remaining steps to QED
GAMMA = 0.95

GENERATION_PROMPT_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}\n```\n"
)

DEFAULT_BENCHMARK_PATH = (
    "/mnt/filesystem-m5/formal/ReProver/data/leandojo_benchmark_4/random/train.json"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEntry:
    state: str
    label: float
    theorem: str
    proved: bool
    response: str

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "label": self.label,
            "theorem": self.theorem,
            "proved": self.proved,
            "response": self.response,
        }


@dataclass
class CollectorStats:
    total_theorems: int = 0
    total_attempts: int = 0
    total_proved: int = 0
    theorems_with_proof: int = 0
    generation_errors: int = 0
    verification_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.total_theorems / elapsed * 60 if elapsed > 0 else 0
        return (
            f"Theorems: {self.total_theorems} | "
            f"Attempts: {self.total_attempts} | "
            f"Proved: {self.total_proved} ({self.total_proved / max(self.total_attempts, 1) * 100:.1f}%) | "
            f"Theorems w/ proof: {self.theorems_with_proof} ({self.theorems_with_proof / max(self.total_theorems, 1) * 100:.1f}%) | "
            f"Gen errors: {self.generation_errors} | "
            f"Verify errors: {self.verification_errors} | "
            f"Rate: {rate:.1f} thm/min"
        )


# ---------------------------------------------------------------------------
# Lean code extraction
# ---------------------------------------------------------------------------

def extract_lean_code(response: str) -> Optional[str]:
    """Extract lean4 code block from a model response.

    Returns the code with Mathlib import prepended if missing, or None if no
    code block is found.
    """
    blocks = re.findall(r"```lean4?\s*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    code = blocks[-1].strip()
    if not code:
        return None
    if "import Mathlib" not in code:
        code = "import Mathlib\n\n" + code
    return code


# ---------------------------------------------------------------------------
# SGLang generation
# ---------------------------------------------------------------------------

async def generate_proofs(
    session: aiohttp.ClientSession,
    sglang_url: str,
    prompt: str,
    n_samples: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate n proof attempts for a single prompt via SGLang.

    Uses the OpenAI-compatible /v1/completions endpoint with n > 1 to get
    multiple samples in a single batched request.

    Returns a list of response strings (may be shorter than n_samples on error).
    """
    payload = {
        "model": "default",
        "prompt": prompt,
        "n": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{sglang_url}/v1/completions"

    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    return [choice["text"] for choice in data["choices"]]


# ---------------------------------------------------------------------------
# Kimina verification
# ---------------------------------------------------------------------------

async def verify_proof(
    session: aiohttp.ClientSession,
    kimina_url: str,
    lean_code: str,
    timeout: int = 60,
) -> bool:
    """Verify a single Lean proof via kimina-lean-server.

    Returns True if the proof compiles without errors.
    """
    payload = {
        "snippets": [{"id": "0", "code": lean_code}],
        "timeout": timeout,
    }

    async with session.post(
        kimina_url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout + 30),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    results = data.get("results", [])
    if not results:
        return False
    response_body = results[0].get("response", {})
    # kimina returns an empty response (or no messages) for valid proofs
    return not response_body or not response_body.get("messages")


async def verify_proofs_batch(
    session: aiohttp.ClientSession,
    kimina_url: str,
    codes: list[Optional[str]],
    concurrency: int = 16,
    timeout: int = 60,
) -> list[bool]:
    """Verify multiple proofs concurrently with bounded parallelism."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[bool] = [False] * len(codes)

    async def _verify(idx: int, code: Optional[str]) -> None:
        if code is None:
            return
        async with semaphore:
            try:
                results[idx] = await verify_proof(session, kimina_url, code, timeout)
            except Exception as e:
                logger.debug("Verification error for index %d: %s", idx, e)

    await asyncio.gather(*[_verify(i, c) for i, c in enumerate(codes)])
    return results


# ---------------------------------------------------------------------------
# Theorem loading
# ---------------------------------------------------------------------------

def load_theorems(
    path: str, num_theorems: Optional[int] = None
) -> list[dict]:
    """Load theorems from the LeanDojo benchmark JSON.

    Each theorem dict is expected to have at least 'full_name' and either
    'formal_statement' or enough information to reconstruct a prompt.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

    if num_theorems is not None and num_theorems > 0:
        data = data[:num_theorems]

    logger.info("Loaded %d theorems from %s", len(data), path)
    return data


def build_generation_prompt(theorem: dict) -> Optional[str]:
    """Build the generation prompt for a theorem.

    Looks for 'formal_statement' first, then falls back to constructing one
    from 'full_name' and header fields common in LeanDojo exports.
    """
    formal = theorem.get("formal_statement")
    if formal:
        return GENERATION_PROMPT_TEMPLATE.format(formal_statement=formal)

    # Fallback: try to build from file content / header info
    # LeanDojo benchmark entries often have 'start', 'end', 'file_path', etc.
    # but not always a standalone formal statement.  Use full_name as a
    # minimal prompt so we can still attempt generation.
    full_name = theorem.get("full_name")
    if full_name:
        return GENERATION_PROMPT_TEMPLATE.format(formal_statement=full_name)

    return None


def format_value_model_state(theorem: dict) -> str:
    """Format the theorem statement for the value model prompt template.

    Uses the formal statement if available, otherwise the full name.
    """
    return theorem.get("formal_statement", theorem.get("full_name", ""))


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

async def collect_trajectories(
    theorems: list[dict],
    sglang_url: str,
    kimina_url: str,
    output_path: str,
    n_samples: int = 8,
    max_tokens: int = 4096,
    temperature: float = 0.8,
    gen_concurrency: int = 4,
    verify_concurrency: int = 16,
) -> CollectorStats:
    """Collect proof search trajectories for all given theorems.

    For each theorem:
      1. Generate n_samples proof attempts via SGLang.
      2. Extract Lean code blocks from responses.
      3. Verify each extracted proof via kimina-lean-server.
      4. Record a trajectory entry per attempt.

    Writes results incrementally to output_path (JSONL).
    """
    stats = CollectorStats()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # We use a single TCP connection pool for the entire run
    connector = aiohttp.TCPConnector(limit=gen_concurrency + verify_concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        gen_semaphore = asyncio.Semaphore(gen_concurrency)

        with open(output_path, "w") as out_f:

            async def process_theorem(idx: int, theorem: dict) -> None:
                prompt = build_generation_prompt(theorem)
                if prompt is None:
                    return

                theorem_name = theorem.get("full_name", f"theorem_{idx}")
                state_text = format_value_model_state(theorem)

                # --- Generation ---
                async with gen_semaphore:
                    try:
                        responses = await generate_proofs(
                            session, sglang_url, prompt,
                            n_samples, max_tokens, temperature,
                        )
                    except Exception as e:
                        logger.warning(
                            "Generation failed for %s: %s", theorem_name, e
                        )
                        stats.generation_errors += 1
                        return

                # --- Code extraction ---
                codes = [extract_lean_code(r) for r in responses]

                # --- Verification (concurrent within this theorem) ---
                try:
                    verified = await verify_proofs_batch(
                        session, kimina_url, codes,
                        concurrency=verify_concurrency,
                    )
                except Exception as e:
                    logger.warning(
                        "Verification batch failed for %s: %s", theorem_name, e
                    )
                    stats.verification_errors += 1
                    verified = [False] * len(codes)

                # --- Record trajectories with gamma^d labels ---
                any_proved = False
                for resp_text, code, is_proved in zip(responses, codes, verified):
                    if is_proved and code:
                        # Count tactic steps in the proof for depth-based label
                        # Each line after ":= by" that isn't blank/comment is a step
                        proof_lines = []
                        in_proof = False
                        for line in code.split("\n"):
                            stripped = line.strip()
                            if ":= by" in line:
                                in_proof = True
                                continue
                            if in_proof and stripped and not stripped.startswith("--"):
                                proof_lines.append(stripped)
                        num_steps = max(len(proof_lines), 1)
                        # Label = gamma^0 = 1.0 for the final state (theorem level)
                        # For the theorem prompt, d = num_steps (distance from start to QED)
                        label = GAMMA ** num_steps
                    else:
                        label = 0.0

                    entry = TrajectoryEntry(
                        state=VALUE_MODEL_PROMPT_TEMPLATE.format(state=state_text),
                        label=round(label, 4),
                        theorem=theorem_name,
                        proved=is_proved,
                        response=resp_text,
                    )
                    out_f.write(json.dumps(entry.to_dict()) + "\n")
                    stats.total_attempts += 1
                    if is_proved:
                        stats.total_proved += 1
                        any_proved = True

                if any_proved:
                    stats.theorems_with_proof += 1

                stats.total_theorems += 1

                if stats.total_theorems % 100 == 0:
                    logger.info(
                        "[Progress %d/%d] %s",
                        stats.total_theorems,
                        len(theorems),
                        stats.summary(),
                    )

            # Process theorems with bounded concurrency.
            # We process them in order so that JSONL output is deterministic
            # given a fixed server, but each theorem's generation + verification
            # pipeline runs concurrently with others up to gen_concurrency.
            tasks: list[asyncio.Task] = []
            for idx, theorem in enumerate(theorems):
                task = asyncio.create_task(process_theorem(idx, theorem))
                tasks.append(task)
                # Limit the number of in-flight theorems to avoid overwhelming
                # memory when the dataset is very large.
                if len(tasks) >= gen_concurrency * 4:
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Propagate exceptions from completed tasks
                    for t in done:
                        if t.exception():
                            logger.error(
                                "Task failed: %s", t.exception()
                            )
                    tasks = list(pending)

            # Await remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        logger.error("Task failed: %s", r)

    # Flush is handled by context manager
    out_f_final_check = Path(output_path)
    if out_f_final_check.exists():
        line_count = sum(1 for _ in open(output_path))
        logger.info(
            "Wrote %d trajectory entries to %s", line_count, output_path
        )

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect proof search trajectories for value model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--theorems",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to LeanDojo benchmark JSON (list of theorem dicts).",
    )
    parser.add_argument(
        "--sglang-url",
        default="http://localhost:30000",
        help="SGLang server URL.",
    )
    parser.add_argument(
        "--kimina-url",
        default="http://localhost:8000/api/check",
        help="kimina-lean-server verification endpoint URL.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for trajectory data.",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=None,
        help="Limit number of theorems to process (default: all).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of proof attempts per theorem.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generation tokens per attempt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--gen-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent generation requests.",
    )
    parser.add_argument(
        "--verify-concurrency",
        type=int,
        default=16,
        help="Maximum concurrent verification requests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Loading theorems from %s", args.theorems)
    theorems = load_theorems(args.theorems, args.num_theorems)

    logger.info(
        "Starting trajectory collection: %d theorems, %d samples each, "
        "sglang=%s, kimina=%s",
        len(theorems),
        args.n_samples,
        args.sglang_url,
        args.kimina_url,
    )

    stats = asyncio.run(
        collect_trajectories(
            theorems=theorems,
            sglang_url=args.sglang_url,
            kimina_url=args.kimina_url,
            output_path=args.output,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gen_concurrency=args.gen_concurrency,
            verify_concurrency=args.verify_concurrency,
        )
    )

    logger.info("Collection complete. %s", stats.summary())


if __name__ == "__main__":
    main()
