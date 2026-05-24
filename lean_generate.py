"""Custom generation function for SLIME: generate tactics for Lean4 proof states."""

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate(args, sample: Sample, sampling_params):
    """
    Generate a tactic for a Lean proof state.

    Input prompt format: [GOAL]\n{proof_state}\n[PROOFSTEP]\n
    Output: a single tactic (stop at newline)
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    # Override sampling params for tactic generation
    gen_params = dict(sampling_params)
    gen_params["max_new_tokens"] = min(gen_params.get("max_new_tokens", 256), 256)

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": gen_params,
        "return_logprob": True,
    }

    output = await post(url, payload)

    # Extract response tokens from logprob info
    response_tokens = [
        item[1] for item in output["meta_info"]["output_token_logprobs"]
    ]

    # Strip at first newline (tactic should be single line)
    response_text = output["text"]
    if "\n" in response_text:
        response_text = response_text.split("\n")[0]
        # Re-tokenize the truncated response
        response_tokens = state.tokenizer.encode(
            response_text, add_special_tokens=False
        )

    sample.tokens = prompt_ids + response_tokens
    sample.response = response_text.strip()
    sample.response_length = len(response_tokens)
    sample.loss_mask = [1] * len(response_tokens)

    finish_reason = output["meta_info"]["finish_reason"]["type"]
    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    # Store logprobs for TIS if available
    if "output_token_logprobs" in output["meta_info"]:
        sample.rollout_log_probs = [
            item[0] for item in output["meta_info"]["output_token_logprobs"]
        ][:sample.response_length]

    return sample
