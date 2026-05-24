"""
Custom SLIME rollout that does value-guided proof search.

During rollout:
1. Generator (DeepSeek-Prover-V2-7B on SGLang) proposes tactics
2. Value model (Llama-3.2-1B on separate SGLang) scores proof states
3. Best-first search uses value scores to prioritize states
4. Kimina-lean-server verifies completed proofs

This replaces the default SLIME rollout with a search-based rollout.
"""

import asyncio
import re
import aiohttp
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


# External services
VALUE_MODEL_URL = "http://localhost:30001"  # SGLang serving Llama-3.2-1B
KIMINA_URL = "http://172.17.0.1:8000/api/check"


async def score_state(state_text: str) -> float:
    """Query the value model LLM for a proof state score.

    The value model predicts gamma^d — discounted distance to QED.
    Values near 1.0 = close to done, near 0.0 = far away or dead end.
    """
    prompt = f"Estimate the discounted distance to proof completion for the following Lean 4 proof state. Output a value between 0.0 (dead end or far from done) and 1.0 (very close to QED).\n\nProof state:\n{state_text}\n\nValue:"

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{VALUE_MODEL_URL}/v1/completions", json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.0,
        }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()

    # Parse score from output
    text = data["choices"][0]["text"].strip()
    try:
        score = float(text.split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        return 0.5  # default if parsing fails


async def verify_proof(lean_code: str) -> bool:
    """Verify a proof with kimina-lean-server."""
    if not lean_code:
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(KIMINA_URL, json={
                "snippets": [{"id": "0", "code": lean_code}],
                "timeout": 60,
            }, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                data = await resp.json()
        results = data.get("results", [])
        if results:
            r = results[0].get("response", {})
            return not r or not r.get("messages")
        return False
    except Exception:
        return False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Custom rollout function for value-guided proof generation.

    The generator produces a complete proof. The value model can optionally
    be used to score intermediate states during multi-turn proof refinement.

    For now: single-turn whole-proof generation (matching current GRPO setup).
    Value model integration comes in the co-training loop.
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    output = await post(url, payload)

    response_tokens = [
        item[1] for item in output["meta_info"]["output_token_logprobs"]
    ]

    sample.tokens = prompt_ids + response_tokens
    sample.response = output["text"]
    sample.response_length = len(response_tokens)
    sample.loss_mask = [1] * len(response_tokens)

    finish_reason = output["meta_info"]["finish_reason"]["type"]
    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    if "output_token_logprobs" in output["meta_info"]:
        sample.rollout_log_probs = [
            item[0] for item in output["meta_info"]["output_token_logprobs"]
        ][:sample.response_length]

    return sample
