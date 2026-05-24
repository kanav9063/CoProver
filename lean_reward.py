"""
Whole-proof reward for SLIME GRPO.
Extract lean4 code block, send to kimina-lean-server, get valid/invalid.
No code block = 0 reward.
"""

import re
import aiohttp

LEAN_SERVER_URL = "http://172.17.0.1:8000/api/check"


async def compute_reward(args, sample) -> float:
    response = sample.response.strip()
    if not response:
        return 0.0

    # Extract lean4 code block — no block = no reward
    blocks = re.findall(r'```lean4?\s*\n(.*?)```', response, re.DOTALL)
    if not blocks:
        return 0.0

    code = blocks[-1].strip()
    if not code:
        return 0.0

    # Add imports if missing
    if "import Mathlib" not in code:
        code = "import Mathlib\n\n" + code

    # Send to kimina-lean-server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LEAN_SERVER_URL,
                json={"snippets": [{"id": "0", "code": code}], "timeout": 60},
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                data = await resp.json()

        results = data.get("results", [])
        if results:
            r = results[0].get("response", {})
            if not r or not r.get("messages"):
                return 1.0
        return 0.0
    except Exception:
        return 0.0
