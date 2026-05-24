"""
Lean verification HTTP server.
Handles tactic verification requests from the SLIME reward function.
Runs outside of Ray to avoid process conflicts.

Usage:
    python lean_server.py --port 8100 --workers 4
"""

import os
import json
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor
from aiohttp import web

os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ.get("PATH", "")
os.environ["DISABLE_REMOTE_CACHE"] = "1"


def verify_tactic_worker(url, commit, file_path, full_name, tactic):
    """Run in a separate process to avoid GIL and Lean process issues."""
    os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ.get("PATH", "")
    os.environ["DISABLE_REMOTE_CACHE"] = "1"

    try:
        from lean_dojo import (
            LeanGitRepo, Theorem, Dojo,
            TacticState, ProofFinished,
            DojoInitError, DojoCrashError,
        )

        repo = LeanGitRepo(url, commit)
        thm = Theorem(repo, file_path, full_name)

        with Dojo(thm, timeout=30) as (dojo, init_state):
            result = dojo.run_tac(init_state, tactic)
            if isinstance(result, ProofFinished):
                return {"reward": 1.0, "status": "proof_finished"}
            elif isinstance(result, TacticState):
                return {"reward": 1.0, "status": "progress", "new_state": result.pp[:200]}
            else:
                return {"reward": 0.0, "status": "error", "error": str(result)[:200]}

    except DojoInitError as e:
        return {"reward": 0.0, "status": "init_error", "error": str(e)[:200]}
    except DojoCrashError as e:
        return {"reward": 0.0, "status": "crash", "error": str(e)[:200]}
    except Exception as e:
        return {"reward": 0.0, "status": "exception", "error": str(e)[:200]}


class LeanServer:
    def __init__(self, num_workers=4):
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.request_count = 0
        self.success_count = 0

    async def handle_verify(self, request):
        data = await request.json()

        url = data.get("url", "")
        commit = data.get("commit", "")
        file_path = data.get("file_path", "")
        full_name = data.get("full_name", "")
        tactic = data.get("tactic", "")

        if not all([url, commit, file_path, full_name, tactic]):
            return web.json_response({"reward": 0.0, "status": "missing_fields"})

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            verify_tactic_worker,
            url, commit, file_path, full_name, tactic
        )

        self.request_count += 1
        if result["reward"] > 0:
            self.success_count += 1

        return web.json_response(result)

    async def handle_health(self, request):
        return web.json_response({
            "status": "ok",
            "requests": self.request_count,
            "successes": self.success_count,
            "rate": self.success_count / max(self.request_count, 1),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    server = LeanServer(num_workers=args.workers)
    app = web.Application()
    app.router.add_post("/verify", server.handle_verify)
    app.router.add_get("/health", server.handle_health)

    print(f"Starting Lean verification server on port {args.port} with {args.workers} workers")
    web.run_app(app, port=args.port)


if __name__ == "__main__":
    main()
