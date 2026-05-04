"""F4 CPU bucket cache — HF-format weight buckets keyed by training step.

The cache_owner rank (pp0 + dp0 + tp0 + cp0 — see F18 / scope F18 cache_owner
uniqueness) builds and stores buckets after every training step the
RLix scheduler asks for. Receivers (SGLang engines) load the buckets
during the F4-F6 selective sync atomic unit driven by
:class:`MilesModelUpdateService`.

Design invariants
-----------------

- **Single ready slot** (``_cache_ready_step``): only the most recently
  built step is exposed for sync. Pipelines do not pin historical
  versions — the M11.1 SLA is base v=-1 plus the latest training step.
  Lookup by any other step raises.
- **Per-bucket payload contains NO ``weight_version``** (scope F21). The
  weight_version is published exactly once per sync via
  ``manager.set_weight_version`` at the end of the atomic unit, never
  per-bucket.
- **HF-format gather** (scope F18): names + shapes + dtypes are HF /
  HuggingFace conventions, not Megatron-internal. The Megatron→HF
  conversion runs upstream in
  :class:`MegatronTrainRayActor.build_cpu_bucket_cache` (iter 11) and
  passes already-converted tensors into :meth:`CPUBucketCache.put`.
- **Cache owner uniqueness**: only the cache_owner rank actually stores
  bucket data. Non-cache_owner ranks instantiate their own
  :class:`CPUBucketCache` for per-rank state, but they MUST NOT call
  ``put`` — the receive-side ``run_sync_session`` body (iter 12) drives
  data transport through the cache_owner only.
- **No per-rank version inversion** (scope F20): publishing a new
  ``_cache_ready_step`` MUST happen inside the same critical section
  that wrote the bucket list. The class itself is single-method-single-
  critical-section friendly: ``put_step`` builds and publishes
  atomically; ``get_step`` is a pure read.
- **Tmpfs naming convention** (scope F66): callers that materialize
  bucket payloads to ``/dev/shm`` for the cpu_serialize transport
  use the format ``miles_cpu_bucket_{uuid}.pt`` so leak detection is
  grep-friendly (``ls /dev/shm | grep miles_cpu_bucket_``).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

# F66 — leak-detection-friendly file naming for the cpu_serialize transport.
TMPFS_FILE_PREFIX = "miles_cpu_bucket_"


@dataclasses.dataclass(frozen=True)
class BucketEntry:
    """A single CPU bucket's metadata + tensors.

    ``params`` is a mapping from HF parameter name -> torch.Tensor on CPU
    (or pinned RAM). The tensors are HF-format (post Megatron→HF
    conversion); receivers load them by name.

    ``size_bytes`` is the post-conversion total payload size used for the
    F10 startup S2 / S3a-2 capacity checks; ``element_count`` is the
    count of scalar elements (debug-only).
    """

    bucket_index: int
    params: dict[str, torch.Tensor]
    size_bytes: int
    element_count: int

    def __post_init__(self) -> None:  # frozen=True allows __post_init__ via object.__setattr__
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative; got {self.size_bytes}")
        for name, tensor in self.params.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"BucketEntry param {name!r} must be torch.Tensor; got {type(tensor).__name__}"
                )
            if tensor.is_cuda:
                raise ValueError(
                    f"BucketEntry param {name!r} must be on CPU (or pinned host RAM); "
                    f"got device={tensor.device}. The cpu_serialize transport requires "
                    f"a CPU tensor before tmpfs materialization."
                )


class CPUBucketCache:
    """Per-rank cache of HF-format CPU buckets keyed by training step.

    Only the cache_owner rank holds non-empty buckets. Other ranks
    instantiate this class but call ``put_empty_step`` so their
    ``_cache_ready_step`` advances in lockstep without retaining bucket
    payloads.

    Thread safety: a single ``threading.Lock`` guards bucket + ready-step
    mutation. The cache_owner builds buckets serially during a training
    step (Megatron->HF gather is collective; the lock serializes
    publishing within the actor process). Read-side lookup
    (:meth:`get_step` / :meth:`is_ready_for`) is also lock-guarded so
    publish/lookup races against ``MilesModelUpdateService`` see a
    consistent snapshot.
    """

    def __init__(self, *, max_bucket_size_bytes: int):
        if max_bucket_size_bytes <= 0:
            raise ValueError(
                f"max_bucket_size_bytes must be positive; got {max_bucket_size_bytes}"
            )
        self._max_bucket_size_bytes: int = int(max_bucket_size_bytes)
        self._lock = threading.Lock()
        # The single-ready-slot pointer (-1 = base from initial INIT, set
        # at init bootstrap Step 4; >=0 = post-training-step build).
        self._cache_ready_step: int | None = None
        # Mapping step -> list[BucketEntry]. Only the most recently
        # published step is retained; older entries are dropped on
        # ``put_step``.
        self._buckets: dict[int, list[BucketEntry]] = {}

    # -- public properties ---------------------------------------------

    @property
    def max_bucket_size_bytes(self) -> int:
        return self._max_bucket_size_bytes

    @property
    def cache_ready_step(self) -> int | None:
        with self._lock:
            return self._cache_ready_step

    def is_ready_for(self, step: int) -> bool:
        """Return True iff the cache currently holds buckets for ``step``."""
        with self._lock:
            return self._cache_ready_step == int(step) and int(step) in self._buckets

    # -- mutation ------------------------------------------------------

    def put_step(self, step: int, buckets: Iterable[BucketEntry]) -> None:
        """Publish a freshly-built bucket list as the new ready step.

        Discards any prior step's data (single-ready-slot invariant).
        Validates that bucket sizes do not exceed
        ``max_bucket_size_bytes`` so the F10 startup capacity check
        (S2 / S3a-2) holds at runtime too.

        Atomic: the bucket-list write and the ``_cache_ready_step``
        advance happen in the same critical section so concurrent
        readers see either the old (step, buckets) tuple or the new
        one — never a torn state.
        """
        bucket_list = list(buckets)
        for entry in bucket_list:
            if entry.size_bytes > self._max_bucket_size_bytes:
                raise ValueError(
                    f"BucketEntry {entry.bucket_index} size {entry.size_bytes} exceeds "
                    f"max_bucket_size_bytes={self._max_bucket_size_bytes}; "
                    f"reduce --miles-model-update-bucket-size-mb or split the bucket."
                )
        step_int = int(step)
        with self._lock:
            self._buckets.clear()
            self._buckets[step_int] = bucket_list
            self._cache_ready_step = step_int
        logger.info(
            "[cpu_bucket_cache] published step=%s buckets=%d total_bytes=%d",
            step_int,
            len(bucket_list),
            sum(e.size_bytes for e in bucket_list),
        )

    def put_empty_step(self, step: int) -> None:
        """Advance the ready-step pointer without retaining buckets.

        Used by non-cache_owner ranks: they participate in the collective
        gather (so the cache_owner can produce HF-format weights) but
        discard the locally-constructed tensors. Their
        ``_cache_ready_step`` still advances in lockstep so the rest of
        the F4-F6 atomic unit can verify cross-rank readiness if it
        wants to.
        """
        step_int = int(step)
        with self._lock:
            self._buckets.clear()
            self._buckets[step_int] = []
            self._cache_ready_step = step_int

    def clear(self) -> None:
        """Drop every retained bucket and reset the ready-step pointer."""
        with self._lock:
            self._buckets.clear()
            self._cache_ready_step = None

    # -- read-side -----------------------------------------------------

    def get_step(self, step: int) -> list[BucketEntry]:
        """Return the bucket list for ``step``.

        Raises :class:`KeyError` if ``step`` is not the currently-ready
        step (single-ready-slot invariant — historical versions are not
        retained). The returned list is a shallow copy; tensor handles
        inside ``BucketEntry`` are shared with the cache (callers MUST
        NOT mutate the tensors).
        """
        step_int = int(step)
        with self._lock:
            if self._cache_ready_step != step_int:
                raise KeyError(
                    f"CPUBucketCache has no buckets for step={step_int}; "
                    f"current ready_step={self._cache_ready_step}. The single-"
                    f"ready-slot invariant means only the most recent step is "
                    f"retrievable."
                )
            return list(self._buckets.get(step_int, []))

    def get_total_bytes(self, step: int) -> int:
        """Sum of ``BucketEntry.size_bytes`` across the ready step."""
        return sum(entry.size_bytes for entry in self.get_step(step))

    # -- transport helpers --------------------------------------------

    @staticmethod
    def make_tmpfs_filename(bucket_index: int, *, suffix: str = ".pt") -> str:
        """F66: produce a leak-detection-friendly tmpfs file name.

        Used by :class:`MilesModelUpdateService` (iter 19) when
        materializing a bucket onto ``/dev/shm`` for the cpu_serialize
        transport. The wrapper code is responsible for ``try/finally
        os.unlink`` of the returned path; the SGLang server-side route
        only reads.
        """
        return f"{TMPFS_FILE_PREFIX}{bucket_index:04d}_{uuid.uuid4().hex}{suffix}"
