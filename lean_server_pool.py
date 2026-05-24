"""
Persistent Lean Server Pool for fast tactic verification during RL training.

Instead of opening a new Lean REPL per verification, we maintain a pool of
pre-warmed Dojo instances. Each Dojo is bound to a specific theorem and can
verify multiple tactic attempts quickly (just JSON send/receive, no process spawn).

Architecture:
    - Pool of worker processes, each managing a set of Dojos
    - Dojos are lazily initialized and cached by theorem key
    - Verification requests are routed to the right Dojo
    - LRU eviction when pool is full
"""

import asyncio
import time
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from loguru import logger

from lean_dojo import (
    LeanGitRepo,
    Theorem,
    Dojo,
    TacticState,
    ProofFinished,
    LeanError,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoTacticTimeoutError,
)


@dataclass
class VerifyResult:
    """Result of tactic verification."""
    reward: float
    new_state: Optional[str] = None  # Pretty-printed new proof state (if progress)
    error: Optional[str] = None


class DojoCache:
    """LRU cache of Dojo instances for a single worker process."""

    def __init__(self, max_size: int = 32, timeout: int = 30):
        self.max_size = max_size
        self.timeout = timeout
        self._cache: OrderedDict[str, Tuple[Dojo, object, object]] = OrderedDict()
        # Maps theorem_key -> (dojo_context_manager, dojo, init_state)

    def _theorem_key(self, url: str, commit: str, file_path: str, full_name: str) -> str:
        return f"{url}@{commit}:{file_path}:{full_name}"

    def _evict_oldest(self):
        """Remove the least recently used Dojo."""
        if self._cache:
            key, (ctx, dojo, _) = self._cache.popitem(last=False)
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            logger.debug(f"Evicted Dojo for {key}")

    def get_or_create(
        self, url: str, commit: str, file_path: str, full_name: str
    ) -> Tuple[object, object]:
        """Get an existing Dojo or create a new one. Returns (dojo, init_state)."""
        key = self._theorem_key(url, commit, file_path, full_name)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            ctx, dojo, init_state = self._cache[key]
            return dojo, init_state

        # Evict if full
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Create new Dojo
        repo = LeanGitRepo(url, commit)
        thm = Theorem(repo, file_path, full_name)
        dojo_obj = Dojo(thm, timeout=self.timeout)
        dojo, init_state = dojo_obj.__enter__()
        self._cache[key] = (dojo_obj, dojo, init_state)
        logger.debug(f"Created Dojo for {key} (cache size: {len(self._cache)})")
        return dojo, init_state

    def close_all(self):
        """Close all cached Dojos."""
        for key, (ctx, dojo, _) in self._cache.items():
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._cache.clear()


# Global cache per worker process
_worker_cache: Optional[DojoCache] = None


def _get_cache() -> DojoCache:
    global _worker_cache
    if _worker_cache is None:
        _worker_cache = DojoCache(max_size=64, timeout=30)
    return _worker_cache


def verify_tactic_sync(
    url: str,
    commit: str,
    file_path: str,
    full_name: str,
    tactic: str,
) -> VerifyResult:
    """
    Verify a tactic using the persistent Dojo cache.
    Runs in a worker process.
    """
    cache = _get_cache()

    try:
        dojo, init_state = cache.get_or_create(url, commit, file_path, full_name)
    except (DojoInitError, Exception) as e:
        return VerifyResult(reward=0.0, error=f"DojoInit: {e}")

    try:
        result = dojo.run_tac(init_state, tactic)

        if isinstance(result, ProofFinished):
            return VerifyResult(reward=1.0, new_state="no goals")
        elif isinstance(result, TacticState):
            return VerifyResult(reward=0.5, new_state=result.pp)
        elif isinstance(result, LeanError):
            return VerifyResult(reward=0.0, error=result.error[:200])
        elif isinstance(result, (DojoTacticTimeoutError, ProofGivenUp)):
            return VerifyResult(reward=0.0, error="timeout/given_up")
        else:
            return VerifyResult(reward=0.0, error=f"unknown: {type(result)}")

    except DojoCrashError as e:
        # Dojo crashed — remove from cache so it gets recreated
        key = cache._theorem_key(url, commit, file_path, full_name)
        if key in cache._cache:
            try:
                ctx, _, _ = cache._cache.pop(key)
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        return VerifyResult(reward=0.0, error=f"crash: {e}")
    except Exception as e:
        return VerifyResult(reward=0.0, error=f"exception: {e}")


class LeanServerPool:
    """
    Async pool of Lean verification workers.

    Usage:
        pool = LeanServerPool(num_workers=8)
        result = await pool.verify(url, commit, file_path, full_name, tactic)
        print(result.reward)  # 0.0, 0.5, or 1.0
    """

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_get_cache,  # Pre-init cache in each worker
        )
        logger.info(f"LeanServerPool started with {num_workers} workers")

    async def verify(
        self,
        url: str,
        commit: str,
        file_path: str,
        full_name: str,
        tactic: str,
    ) -> VerifyResult:
        """Verify a tactic asynchronously using the worker pool."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            verify_tactic_sync,
            url, commit, file_path, full_name, tactic,
        )
        return result

    async def verify_batch(
        self,
        requests: list[dict],
    ) -> list[VerifyResult]:
        """Verify multiple tactics concurrently."""
        tasks = [
            self.verify(
                r["url"], r["commit"], r["file_path"], r["full_name"], r["tactic"]
            )
            for r in requests
        ]
        return await asyncio.gather(*tasks)

    def shutdown(self):
        """Shutdown the pool."""
        self.executor.shutdown(wait=True)
        logger.info("LeanServerPool shutdown complete")


# Singleton pool
_global_pool: Optional[LeanServerPool] = None


def get_lean_pool(num_workers: int = 8) -> LeanServerPool:
    """Get or create the global Lean server pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = LeanServerPool(num_workers=num_workers)
    return _global_pool
