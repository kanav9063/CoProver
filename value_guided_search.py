"""
Value-Guided Best-First Search.

Extends ReProver's BestFirstSearchProver to use a learned value model
for prioritizing proof states instead of cumulative log-probability.
"""

import sys
import os
import torch
import asyncio
from typing import Optional, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ReProver"))

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp, DojoTacticTimeoutError
from prover.proof_search import BestFirstSearchProver
from prover.search_tree import InternalNode, ProofFinishedNode, ErrorNode, Edge, Status
from value_model import ValueModelServer


class ValueGuidedSearchProver(BestFirstSearchProver):
    """
    Best-first search where states are prioritized by a learned value model
    instead of cumulative log-probability.

    priority = alpha * cumulative_logprob + (1 - alpha) * value_score
    """

    def __init__(
        self,
        tac_gen,
        timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
        debug: bool,
        value_server: ValueModelServer,
        alpha: float = 0.3,  # weight for logprob (0 = pure value, 1 = pure logprob)
    ):
        super().__init__(tac_gen, timeout, max_expansions, num_sampled_tactics, debug)
        self.value_server = value_server
        self.alpha = alpha
        self._value_cache = {}

    def _get_value(self, state_text: str) -> float:
        """Get value score for a proof state, with caching."""
        if state_text not in self._value_cache:
            scores = self.value_server.score([state_text])
            self._value_cache[state_text] = scores[0]
        return self._value_cache[state_text]

    def _run_tactic(self, node, tactic, logprob, priority_queue):
        """Override to score new states with value model."""
        t0 = __import__("time").time()
        response = self.dojo.run_tac(node.state, tactic)
        elapsed = __import__("time").time() - t0
        self.environment_time += elapsed

        try:
            result_node = self.nodes[response]
        except KeyError:
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (LeanError, DojoTacticTimeoutError, ProofGivenUp):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            # Score with value model if it's a new internal node
            if isinstance(result_node, InternalNode):
                state_text = response.pp if hasattr(response, "pp") else str(response)
                value_score = self._get_value(state_text)

                # Combined priority: alpha * logprob + (1-alpha) * value
                combined_priority = (
                    self.alpha * result_node.cumulative_logprob
                    + (1 - self.alpha) * value_score
                )
                # Override priority by monkey-patching (InternalNode.priority is a property)
                result_node._value_priority = combined_priority

            if result_node.status == Status.OPEN:
                if hasattr(result_node, "_value_priority"):
                    priority_queue.put_nowait((-result_node._value_priority, result_node))
                else:
                    priority_queue.put_nowait((-result_node.priority, result_node))

        self.nodes[response] = result_node
        edge = Edge(tactic=tactic, src=node, dst=result_node)
        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)
