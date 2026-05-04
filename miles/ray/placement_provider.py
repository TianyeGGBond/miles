"""F12 placement provider — adapter between RLix-declared placements and the
per-worker view MILES needs.

The standalone path uses :func:`miles.ray.placement_group.create_placement_groups`
which builds Ray placement groups directly from MILES args. The RLix path
delegates allocation to the RLix scheduler (via a
``RollResourceManagerProxy`` injected by the coordinator), then translates
the resulting allocation into per-worker placements MILES code expects.

Design notes (per scope F33 / F102 / F35):

- ``WorkerPlacement`` is per-worker and node-local: ``placement_group`` is
  a Ray PlacementGroup; ``bundle_index`` indexes within the PG;
  ``gpu_ids`` is a tuple of node-local GPU ids the worker may claim.
  Multi-node-compatible structurally (no global GPU id assumption).
- ``MilesPlacementProvider`` is constructed with declared train and
  infer device mappings (NOT computed inside the provider — they come
  from the F8 driver's ``cluster_device_mappings`` so multi-pipeline
  scenarios don't double-allocate).
- ``get_all_rollout_engine_placements()`` returns the FULL engine table
  (length == ``rollout_num_gpus // rollout_num_gpus_per_engine``). Init
  bootstrap (iter 26) creates every engine; runtime expand activates
  subsets.
- F33: SGLang ``base_gpu_id`` MUST be 0 in RLix mode. CVD is set per
  worker (manual via Ray runtime_env), so post-CVD the process sees
  ``cuda:0..tp-1`` regardless of physical ids. The placement provider
  records physical ids in ``WorkerPlacement.gpu_ids`` for diagnostics
  and CVD construction; SGLang itself reads ``base_gpu_id=0``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerPlacement:
    """Per-worker view of a Ray placement group bundle.

    ``placement_group`` is the Ray PG (opaque object — only Ray APIs
    interpret it). ``bundle_index`` is the logical bundle index within
    the PG, used by ``PlacementGroupSchedulingStrategy(
    placement_group=pg, placement_group_bundle_index=...)``.
    ``gpu_ids`` is a tuple of node-local GPU ids. ``node_rank`` is the
    Ray node rank (0 for single-node).

    Frozen / hashable so the provider can return immutable views and
    callers can use placements as dict keys for engine_index lookups.
    """

    placement_group: object
    bundle_index: int
    gpu_ids: tuple[int, ...]
    node_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_ids, tuple):
            raise TypeError("gpu_ids must be a tuple (frozen)")
        if any(g < 0 for g in self.gpu_ids):
            raise ValueError(f"gpu_ids must be non-negative; got {self.gpu_ids}")
        if list(self.gpu_ids) != sorted(self.gpu_ids):
            raise ValueError(
                f"gpu_ids must be sorted (first-build contiguous-mapping invariant, "
                f"per scope F35 startup structural assert); got {self.gpu_ids}"
            )


class MilesPlacementProvider:
    """Adapter from RLix-declared mappings to MILES per-worker placements.

    Constructed by the F8 driver / coordinator with:
      - ``resource_manager_proxy``: opaque proxy delegating allocation to
        the RLix scheduler. Iter 14 keeps it as ``Any`` (the concrete
        type lives in the rlix package). Provider does NOT instantiate
        a new proxy — the caller injects one (per F12 forbidden:
        provider must NOT self-construct the manager).
      - ``train_device_mapping``: list of physical GPU ids the train
        actors will claim (length ==
        ``actor_num_nodes * actor_num_gpus_per_node``).
      - ``infer_device_mapping``: list of physical GPU ids the inference
        engines will claim (length == ``rollout_num_gpus``).
      - ``rollout_num_gpus_per_engine``: tp_size for each engine.
      - ``num_gpus_per_node``: declared.

    The mappings are passed in (not derived) so multi-pipeline
    scenarios can have different declared mappings without the
    provider re-deriving per-pipeline conflicts.
    """

    def __init__(
        self,
        *,
        resource_manager_proxy,
        train_device_mapping: list[int],
        infer_device_mapping: list[int],
        rollout_num_gpus_per_engine: int,
        num_gpus_per_node: int,
    ):
        if rollout_num_gpus_per_engine <= 0:
            raise ValueError(
                f"rollout_num_gpus_per_engine must be > 0; got {rollout_num_gpus_per_engine}"
            )
        if num_gpus_per_node <= 0:
            raise ValueError(f"num_gpus_per_node must be > 0; got {num_gpus_per_node}")
        if len(infer_device_mapping) % rollout_num_gpus_per_engine != 0:
            raise ValueError(
                f"infer_device_mapping ({infer_device_mapping}) must divide evenly by "
                f"rollout_num_gpus_per_engine ({rollout_num_gpus_per_engine}); "
                f"this mirrors the F10 C6 startup assert."
            )
        # First-build contiguous-mapping structural assert (scope F35).
        # Non-contiguous / custom-ordered mappings need an explicit
        # scheduler_dp_rank -> engine_index adapter (A18 / F95) that is
        # use-case-triggered, not part of M11.1.
        if list(infer_device_mapping) != sorted(infer_device_mapping):
            raise ValueError(
                f"first build requires sorted infer_device_mapping; got "
                f"{infer_device_mapping}. Non-contiguous mapping requires the "
                f"A18 / F95 scheduler_dp_rank adapter (follow-up)."
            )
        # First-build also requires GAP-free (gpu_ids in each engine slice
        # must be a contiguous integer run). E.g. infer_device_mapping=[0, 2]
        # with tp=2 is sorted but the slice (0, 2) is not contiguous.
        for engine_idx in range(len(infer_device_mapping) // int(rollout_num_gpus_per_engine)):
            start = engine_idx * int(rollout_num_gpus_per_engine)
            slice_ids = infer_device_mapping[start : start + int(rollout_num_gpus_per_engine)]
            expected = list(range(slice_ids[0], slice_ids[0] + int(rollout_num_gpus_per_engine)))
            if list(slice_ids) != expected:
                raise ValueError(
                    f"first build requires gap-free GPU ids per engine; engine "
                    f"{engine_idx} got {slice_ids}, expected {expected}. Non-"
                    f"contiguous mapping requires the A18 / F95 adapter."
                )

        self._proxy = resource_manager_proxy
        self._train_device_mapping = list(train_device_mapping)
        self._infer_device_mapping = list(infer_device_mapping)
        self._per_engine = int(rollout_num_gpus_per_engine)
        self._num_gpus_per_node = int(num_gpus_per_node)

    @property
    def engine_count(self) -> int:
        return len(self._infer_device_mapping) // self._per_engine

    def get_all_rollout_engine_placements(self) -> list[WorkerPlacement]:
        """Full engine table — length == :attr:`engine_count`.

        Independent of any runtime-allocated subset: full INIT (iter 26)
        creates every engine, then runtime grants wake selected indices.
        Each engine's ``gpu_ids`` is the contiguous slice of
        ``infer_device_mapping`` covering its tp_size GPUs.

        The Ray PlacementGroup itself is requested from
        ``resource_manager_proxy.allocate_placement_group`` so the
        scheduler can satisfy multi-pipeline allocation policy. F35
        startup structural asserts run in :meth:`assert_structural`.
        """
        engine_count = self.engine_count
        # Ask the proxy for a PG covering the inference pool; the proxy
        # is responsible for honoring the declared infer_device_mapping
        # so the bundle_index sequence corresponds 1:1 to
        # infer_device_mapping ordering.
        pg = self._proxy.allocate_placement_group(
            world_size=len(self._infer_device_mapping),
            device_mapping=tuple(self._infer_device_mapping),
        )
        placements: list[WorkerPlacement] = []
        for engine_idx in range(engine_count):
            start = engine_idx * self._per_engine
            slice_gpu_ids = tuple(self._infer_device_mapping[start : start + self._per_engine])
            # F102: derive node_rank from the first physical GPU id of
            # this slice. num_gpus_per_node tells us node boundaries on
            # a homogeneous cluster. For multi-node deployments the
            # proxy is expected to pin each engine slice to a single
            # node; here we just record the rank for downstream
            # placement-group bundle selection.
            node_rank = slice_gpu_ids[0] // self._num_gpus_per_node
            placements.append(
                WorkerPlacement(
                    placement_group=pg,
                    bundle_index=start,
                    gpu_ids=slice_gpu_ids,
                    node_rank=int(node_rank),
                )
            )
        return placements

    def get_active_engine_indices(
        self,
        allocated_gpus: Iterable[int],
        tp_size: int,
    ) -> frozenset[int]:
        """Map a runtime-allocated GPU set back to engine indices.

        Used by the coordinator's runtime-expand path (iter 23) to
        translate ``scheduler.allocate_inference_resources`` results
        into engine indices the manager can wake. Each engine's
        tp_size GPUs must be either fully present or fully absent in
        ``allocated_gpus`` — partial allocation is invalid.
        """
        if int(tp_size) != self._per_engine:
            raise ValueError(
                f"tp_size mismatch: provider was constructed with "
                f"rollout_num_gpus_per_engine={self._per_engine}, got tp_size={tp_size}"
            )
        allocated = set(int(g) for g in allocated_gpus)
        active: set[int] = set()
        for engine_idx in range(self.engine_count):
            start = engine_idx * self._per_engine
            slice_gpu_ids = set(self._infer_device_mapping[start : start + self._per_engine])
            in_allocated = slice_gpu_ids & allocated
            if not in_allocated:
                continue
            if in_allocated != slice_gpu_ids:
                raise ValueError(
                    f"partial-engine allocation for engine_index={engine_idx}: "
                    f"declared GPUs={sorted(slice_gpu_ids)}, allocated subset="
                    f"{sorted(in_allocated)}. Schedulers must allocate full "
                    f"engine slices."
                )
            active.add(engine_idx)
        return frozenset(active)

    def get_train_workers(self) -> list[WorkerPlacement]:
        """Per-worker placements for the train pool.

        Calls
        ``resource_manager_proxy.allocate_placement_group(world_size=...,
        device_mapping=...)`` once and slices the resulting PG into
        per-worker bundles (one bundle per train GPU).
        """
        pg = self._proxy.allocate_placement_group(
            world_size=len(self._train_device_mapping),
            device_mapping=tuple(self._train_device_mapping),
        )
        placements: list[WorkerPlacement] = []
        for idx, gpu_id in enumerate(self._train_device_mapping):
            placements.append(
                WorkerPlacement(
                    placement_group=pg,
                    bundle_index=idx,
                    gpu_ids=(int(gpu_id),),
                    node_rank=int(gpu_id) // self._num_gpus_per_node,
                )
            )
        return placements

    def assert_structural(self, placements: list[WorkerPlacement]) -> None:
        """F35 startup structural asserts on a returned engine table.

        Verifies length == engine_count, each placement covers exactly
        ``rollout_num_gpus_per_engine`` GPUs, and the slice is
        contiguous starting at the expected stride.
        """
        if len(placements) != self.engine_count:
            raise RuntimeError(
                f"expected {self.engine_count} placements; got {len(placements)}"
            )
        for engine_idx, wp in enumerate(placements):
            if len(wp.gpu_ids) != self._per_engine:
                raise RuntimeError(
                    f"engine_index={engine_idx}: expected {self._per_engine} GPUs, "
                    f"got {wp.gpu_ids}"
                )
            expected_start = engine_idx * self._per_engine
            expected = tuple(
                self._infer_device_mapping[expected_start : expected_start + self._per_engine]
            )
            if tuple(wp.gpu_ids) != expected:
                raise RuntimeError(
                    f"engine_index={engine_idx}: expected gpu_ids={expected}, "
                    f"got {wp.gpu_ids}; first-build contiguous-mapping invariant"
                )


__all__ = ["WorkerPlacement", "MilesPlacementProvider"]
