"""F8 / F9 / F11 RLix entry driver — `examples/rlix/run_miles_rlix.py`.

Standalone entry stays at ``train_async.py``; this script is the RLix-mode
entry. ``RLIX_CONTROL_PLANE=rlix`` MUST be set before any heavy import
(``import torch`` / ``import sglang``) resolves so per-actor CVD can be
honored cleanly via Ray ``runtime_env`` (scope F08 — transitive imports
are the hazard, not first-line imports).

Per scope F13 the driver MUST NOT have a top-level ``try / except`` and
MUST NOT call ``ray.shutdown()``: failure semantics = let exceptions
propagate naturally → driver exits → user runs ``ray stop`` to clean up.
This is intentional minimalism: orchestrator-driven multi-pipeline
cleanup is M11.5 follow-up (F81).

KNOWN-GAP STUB (R07-F1) — this iter-16 surface establishes the env-var
guard + F10 startup invariant + cluster_device_mappings derivation only.
The orchestrator / coordinator / pipeline construction (orchestrator
``request_gpus(actor_train + actor_infer)`` + ``MilesCoordinator`` actor
+ ``coordinator.create_pipeline_actor.remote(...)`` +
``pipeline.initialize_pipeline()`` + main loop) is intentionally
deferred to a future driver iter that wires the full RLix-side scheduler
plumbing. Until that iter lands, invoking this script as
``python -m examples.rlix.run_miles_rlix`` only validates topology and
prints the derived mappings — no GPUs are allocated, no actors are
created, no end-to-end run happens. M11.1 + M11.2 GPU smoke must wait
for the wiring iter.
"""

from __future__ import annotations

import os
import sys

# F08 / F41 — fail fast if RLix entry is invoked without the env var.
# The check must happen BEFORE any heavy import (torch / sglang /
# megatron) so CVD has a chance to take effect via Ray runtime_env.
if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
    sys.stderr.write(
        "examples/rlix/run_miles_rlix.py requires RLIX_CONTROL_PLANE=rlix.\n"
        "Use train_async.py for standalone runs, or set the env var:\n"
        "    RLIX_CONTROL_PLANE=rlix python -m examples.rlix.run_miles_rlix ...\n"
    )
    sys.exit(2)


def _build_cluster_device_mappings(args) -> dict[str, list[int]]:
    """F8 driver — derive cluster_device_mappings from existing args.

    First-build contiguous-mapping invariant (F35 / C6): train pool is
    ``range(actor_num_nodes * actor_num_gpus_per_node)``; infer pool is
    ``range(rollout_num_gpus)``. Both are zero-based shared (RLix mode
    convention) so train can be a strict subset of infer (partial
    overlap topology). No new device_mapping CLI args are introduced
    (Layer 1 forbidden).
    """
    actor_count = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
    rollout_count = int(args.rollout_num_gpus)
    return {
        "actor_train": list(range(actor_count)),
        "actor_infer": list(range(rollout_count)),
    }


def main():
    """RLix entry. Imports heavy modules lazily so the env-var guard above
    fires before transitive ``import torch`` / ``import sglang``.
    """
    # Lazy imports — must NOT be at module top so the env-var guard
    # above fires first under transitive resolution.
    import asyncio  # noqa: F401  -- kept for forward-compat with async main

    from miles.utils.arguments import parse_args
    from miles.utils.logging_utils import configure_logger
    from miles.utils.rlix_validation import assert_rlix_topology

    configure_logger()
    args = parse_args()

    # F10 startup fail-fast — verify partial overlap topology + transport
    # constraints BEFORE allocating any GPUs. R08-F1: pass
    # ``args.sglang_config`` so C9 (PD-disaggregation forbidden) fires
    # at the entry path. ``assert_rlix_topology`` also has an internal
    # fallback to ``args.sglang_config`` when the kwarg is None, so this
    # line is belt-and-suspenders.
    assert_rlix_topology(args, sglang_config=getattr(args, "sglang_config", None))

    # M11.1 driver wiring (orchestrator allocate / register / admit +
    # MilesCoordinator + MilesPipeline) lands as RLix iters 17-27 +
    # iter 23 create_pipeline_actor. Iter 16 only establishes the entry
    # surface and the F10 startup invariant; full wiring happens once
    # the RLix-side modules exist.
    cluster_device_mappings = _build_cluster_device_mappings(args)
    print(
        "[run_miles_rlix] F10 startup validation passed; "
        f"cluster_device_mappings={cluster_device_mappings}"
    )
    print(
        "[run_miles_rlix] Iter 16 entry stub. RLix coordinator + pipeline "
        "wiring lands in iters 21-27. Re-run after those iters land."
    )


if __name__ == "__main__":
    main()
