"""F78 — DEV-ONLY-MVP scaffolding for F4–F6 (cache build + atomic sync + version).

SCAFFOLDING ONLY — these tests do NOT execute as Gate acceptance evidence.
They land as a starting point for the post-coding GPU smoke run; logic
may need iteration once the tests actually run against real engines.

F4 — CPU bucket cache module + cache_owner build + run_sync_session
     single composite RPC + SGLang receiver methods + HTTP route
     injection.
F5 + F6 — Atomic sync unit (transport + finalize + version publish
     under one timeout) + base v=-1 path.

All tests use mocks so the file imports / parses cleanly on a CPU box.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


class TestCPUBucketCache(unittest.TestCase):
    """F4a — single-ready-slot cache invariants."""

    def _module(self):
        from miles.backends.megatron_utils.update_weight import cpu_bucket_cache

        return cpu_bucket_cache

    def test_put_step_replaces_prior_step(self):
        m = self._module()
        cache = m.CPUBucketCache(max_bucket_size_bytes=1024 * 1024)
        b0 = m.BucketEntry(
            bucket_index=0,
            params={"foo": torch.zeros(2, 2)},
            size_bytes=16,
            element_count=4,
        )
        b1 = m.BucketEntry(
            bucket_index=0,
            params={"bar": torch.zeros(2, 2)},
            size_bytes=16,
            element_count=4,
        )
        cache.put_step(0, [b0])
        self.assertEqual(cache.cache_ready_step, 0)
        cache.put_step(1, [b1])
        self.assertEqual(cache.cache_ready_step, 1)
        # Lookup at a stale step raises (single-ready-slot invariant).
        with self.assertRaises(KeyError):
            cache.get_step(0)
        # Current step lookup returns a shallow copy of the bucket list.
        self.assertEqual(len(cache.get_step(1)), 1)

    def test_put_empty_step_advances_pointer_without_data(self):
        m = self._module()
        cache = m.CPUBucketCache(max_bucket_size_bytes=1024 * 1024)
        cache.put_empty_step(7)
        self.assertEqual(cache.cache_ready_step, 7)
        self.assertEqual(cache.get_step(7), [])

    def test_bucket_entry_rejects_cuda_tensor(self):
        m = self._module()
        # Construct a plain CPU tensor — we cannot allocate CUDA in
        # the test env. The validation path is exercised by passing a
        # tensor whose .is_cuda would be True; emulate via mock.
        cuda_tensor = mock.Mock(spec=torch.Tensor)
        cuda_tensor.is_cuda = True
        cuda_tensor.device = torch.device("cuda:0")
        with self.assertRaises(ValueError):
            m.BucketEntry(
                bucket_index=0,
                params={"x": cuda_tensor},
                size_bytes=16,
                element_count=4,
            )


class TestSyncSessionPlanWireShape(unittest.TestCase):
    """F4 — SyncSessionPlan crosses the Ray boundary as a plain dict.

    Tests that as_wire_dict produces every required key the MILES
    side validates (mirrors run_sync_session plan-key validator).
    """

    def test_as_wire_dict_contains_required_keys(self):
        from rlix.pipeline.miles_model_update_service import SyncSessionPlan

        plan = SyncSessionPlan(
            sync_id="test-sync",
            version=-1,
            group_name="miles_test_sync",
            master_addr="127.0.0.1",
            master_port=20001,
            timeout_s=150.0,
            target_handles={0: mock.Mock(), 1: mock.Mock()},
            cpu_serialize_local_ranks=frozenset({0, 1}),
            broadcast_local_ranks=frozenset(),
            comm_ranks={0: 0, 1: 0},
            world_size=1,
        )
        wire = plan.as_wire_dict()
        for key in (
            "sync_id",
            "version",
            "group_name",
            "master_addr",
            "master_port",
            "timeout_s",
            "target_handles",
            "cpu_serialize_local_ranks",
            "broadcast_local_ranks",
            "comm_ranks",
            "world_size",
        ):
            self.assertIn(key, wire, f"missing required key {key!r}")
        # frozenset → list conversion.
        self.assertIsInstance(wire["cpu_serialize_local_ranks"], list)
        self.assertIsInstance(wire["broadcast_local_ranks"], list)


class TestRLixHooksProtocol(unittest.TestCase):
    """F9 — MilesRLixHooks publishes ProgressReport via the canonical RPC."""

    def test_begin_progress_batch_publishes_report(self):
        from rlix.pipeline.miles_hooks import MilesRLixHooks

        coordinator = mock.Mock()
        coordinator.report_progress_from_scheduler = mock.Mock()
        coordinator.report_progress_from_scheduler.remote = mock.Mock()
        hooks = MilesRLixHooks(coordinator, pipeline_id="test")
        hooks.begin_progress_batch(
            target_weight_version=0,
            step_target_groups=4,
            initial_completed=0,
        )
        coordinator.report_progress_from_scheduler.remote.assert_called_once()

    def test_end_progress_batch_calls_clear_stream(self):
        from rlix.pipeline.miles_hooks import MilesRLixHooks

        coordinator = mock.Mock()
        coordinator.clear_progress_stream = mock.Mock()
        coordinator.clear_progress_stream.remote = mock.Mock()
        hooks = MilesRLixHooks(coordinator, pipeline_id="test")
        # Open a batch so end_progress_batch has a stream identity to
        # retire.
        coordinator.report_progress_from_scheduler = mock.Mock()
        coordinator.report_progress_from_scheduler.remote = mock.Mock()
        hooks.begin_progress_batch(
            target_weight_version=0,
            step_target_groups=4,
            initial_completed=0,
        )
        hooks.end_progress_batch()
        coordinator.clear_progress_stream.remote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
