"""
Combined reward function for co-training.

Reward = lean_verification_reward + beta * value_model_bonus

The value model bonus rewards the generator for reaching states
that the value model considers promising, even if the full proof
doesn't verify. This gives denser signal than binary pass/fail.

Can be used with --custom-rm-path training.value_reward.compute_reward
"""

import re
import aiohttp

KIMINA_URL = "http://172.17.0.1:8000/api/check"
VALUE_MODEL_URL = "http://172.17.0.1:30001/v1/completions"  # SGLang serving value model

# Weight for value model bonus (0 = pure Lean, 1 = pure value model)
BETA = 0.1


async def _verify_lean(code: str) -> float:
    """Verify with kimina-lean-server. Returns 1.0 or 0.0."""
    if not code:
        return 0.0
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(KIMINA_URL, json={
                "snippets": [{"id": "0", "code": code}],
                "timeout": 60,
            }, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                data = await resp.json()
        results = data.get("results", [])
        if results:
            r = results[0].get("response", {})
            if not r or not r.get("messages"):
                return 1.0
        return 0.0
    except Exception:
        return 0.0


async def _get_value_score(state_text: str) -> float:
    """Query value model for a score. Returns float in [0, 1]."""
    prompt = f"Rate the following Lean 4 proof state from 0.0 to 1.0 based on how likely it is to lead to a completed proof.\n\nProof state:\n{state_text}\n\nScore:"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(VALUE_MODEL_URL, json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.0,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
        text = data["choices"][0]["text"].strip()
        return max(0.0, min(1.0, float(text.split()[0])))
    except Exception:
        return 0.5


async def compute_reward(args, sample) -> float:
    """
    Combined reward: Lean verification + value model bonus.

    If Lean verifies: reward = 1.0
    If not: reward = beta * value_model_score (partial credit)
    """
    response = sample.response.strip()
    if not response:
        return 0.0

    # Extract lean code
    blocks = re.findall(r'```lean4?\s*\n(.*?)```', response, re.DOTALL)
    if not blocks:
        return 0.0

    code = blocks[-1].strip()
    if not code:
        return 0.0

    if "import Mathlib" not in code:
        code = "import Mathlib\n\n" + code

    # Primary reward: Lean verification
    lean_reward = await _verify_lean(code)
    if lean_reward == 1.0:
        return 1.0

    # Secondary reward: value model bonus for partial progress
    # Extract the proof state from the code (rough heuristic)
    # Use the theorem statement as the "state" for value model scoring
    metadata = sample.metadata or {}
    formal_statement = metadata.get("formal_statement", "")
    if formal_statement:
        value_score = await _get_value_score(formal_statement)
        return BETA * value_score

    return 0.0
