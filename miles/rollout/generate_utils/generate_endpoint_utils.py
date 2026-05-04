"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # TODO shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
    }
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload, None


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    if args.use_miles_router and "RadixTreeMiddleware" in args.miles_router_middleware_paths:
        from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        # TODO may rename to match
        await postprocess_sample_with_radix_tree(args, sample, output)

        assert not update_loss_mask, "This code branch has not implemented update_loss_mask"
    else:
        if x := output["meta_info"].get("output_token_logprobs"):
            new_response_tokens = [item[1] for item in x]
            new_response_log_probs = [item[0] for item in x]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        if update_loss_mask:
            if sample.loss_mask is None:
                sample.loss_mask = []
            sample.loss_mask += [1] * len(new_response_tokens)

    # TODO handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, output, sample, "routed_experts")

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def get_rollout_topk_from_response(args, output, sample, key):
    info = output["meta_info"].get(key)
    if info is None:
        return None
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    return x.reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)


# ---------------------------------------------------------------------------
# F3 turn-level redispatch helpers — snapshot/restore the sample state at
# the boundary of a single multi_turn turn so a scheduler-preempt response
# can be retried against a different engine without poisoning the sample.
# ---------------------------------------------------------------------------


def _snapshot_turn_state(sample: Sample, multi_samples: list[Sample]) -> dict[str, Any]:
    """Capture every sample / multi_samples field mutated during one turn.

    :func:`_restore_turn_state` rolls these back after a preempted turn so
    the redispatch attempt against a different engine starts from the
    same pre-turn state.

    Sample-immutable fields (``prompt``, ``multimodal_inputs``) are not
    captured — they are not mutated by ``update_sample_from_response``.
    """
    return {
        "tokens": list(sample.tokens) if sample.tokens is not None else None,
        "response": sample.response,
        "response_length": sample.response_length,
        "rollout_log_probs": (
            list(sample.rollout_log_probs) if sample.rollout_log_probs is not None else None
        ),
        "loss_mask": list(sample.loss_mask) if sample.loss_mask is not None else None,
        "status": sample.status,
        "rollout_routed_experts": deepcopy(getattr(sample, "rollout_routed_experts", None)),
        "multi_samples_len": len(multi_samples),
    }


def _restore_turn_state(
    sample: Sample, multi_samples: list[Sample], snapshot: dict[str, Any]
) -> None:
    """Inverse of :func:`_snapshot_turn_state`.

    Truncates ``multi_samples`` back to its pre-turn length and rolls back
    the mutable fields on ``sample``. The caller (multi_turn.generate)
    rebuilds the request payload after restore; the snapshot only covers
    per-turn response state.
    """
    sample.tokens = list(snapshot["tokens"]) if snapshot["tokens"] is not None else None
    sample.response = snapshot["response"]
    sample.response_length = snapshot["response_length"]
    sample.rollout_log_probs = (
        list(snapshot["rollout_log_probs"])
        if snapshot["rollout_log_probs"] is not None
        else None
    )
    sample.loss_mask = list(snapshot["loss_mask"]) if snapshot["loss_mask"] is not None else None
    sample.status = snapshot["status"]
    sample.rollout_routed_experts = deepcopy(snapshot["rollout_routed_experts"])
    target_len = int(snapshot["multi_samples_len"])
    if len(multi_samples) > target_len:
        del multi_samples[target_len:]
