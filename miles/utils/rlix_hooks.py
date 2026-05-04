"""RLix integration seam — protocol + no-op default.

MILES code that needs to publish progress, route preempt notifications, or otherwise
talk to the RLix coordinator imports the ``RLixHooks`` Protocol from this module
instead of importing concrete RLix types. Standalone (non-RLix) MILES runs receive a
``NoOpRLixHooks()`` instance so the call sites stay branchless.

The concrete implementation (``MilesRLixHooks``) lives on the RLix side
(``rlix.pipeline.miles_hooks``) and is injected into the RLix entry driver
(``examples/rlix/run_miles_rlix.py``). MILES MUST NOT import from ``rlix.*`` — the
seam is unidirectional.

Forward-compat note (X1 / F108): ``mode`` and ``adapter_id`` are nullable kwargs on
``begin_progress_batch`` to avoid a protocol-signature break when M11.4 LoRA support
introduces multi-adapter progress streams.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RLixHooks(Protocol):
    """Hook surface that the RLix pipeline coordinator implements.

    Any caller (typically ``examples/fully_async/fully_async_rollout.py``) holds a
    reference of this protocol type. Standalone MILES runs use ``NoOpRLixHooks``.
    """

    def begin_progress_batch(
        self,
        target_weight_version: int,
        step_target_groups: int,
        initial_completed: int,
        *,
        mode: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        """Open a progress-reporting batch for a single training step.

        ``mode`` and ``adapter_id`` are reserved for future M11.4 LoRA / multi-stream
        support; current callers pass ``None``.
        """

    def bump_completed(self, *, target_weight_version: int) -> None:
        """Report one completed group toward the active progress batch."""

    def end_progress_batch(self) -> None:
        """Close the progress batch opened by ``begin_progress_batch``."""

    def report_preempt(self, engine_index: int, worker_url: str) -> None:
        """Optional: notify the coordinator that an engine was preempted mid-generate."""


class NoOpRLixHooks:
    """Default standalone implementation — every method is a no-op."""

    def begin_progress_batch(
        self,
        target_weight_version: int,
        step_target_groups: int,
        initial_completed: int,
        *,
        mode: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        return None

    def bump_completed(self, *, target_weight_version: int) -> None:
        return None

    def end_progress_batch(self) -> None:
        return None

    def report_preempt(self, engine_index: int, worker_url: str) -> None:
        return None


__all__ = ["RLixHooks", "NoOpRLixHooks"]
