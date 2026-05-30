from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest import mock


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeEngine:
    def __init__(self, name: str, calls: list[str], on_disable=None, on_shutdown=None):
        self.name = name

        def disable():
            calls.append(f"{name}:disable")
            if on_disable is not None:
                on_disable()

        def shutdown():
            calls.append(f"{name}:shutdown")
            if on_shutdown is not None:
                on_shutdown()

        self.unregister_from_router = _RemoteMethod(disable)
        self.shutdown = _RemoteMethod(shutdown)


def _load_health_monitor_with_fake_ray(calls: list[str]):
    ray_stub = types.ModuleType("ray")

    def fake_get(result, timeout=None):
        calls.append(f"ray.get:{timeout}")
        return result

    def fake_kill(engine):
        calls.append(f"{engine.name}:ray.kill")

    ray_stub.get = fake_get
    ray_stub.kill = fake_kill

    with mock.patch.dict(sys.modules, {"ray": ray_stub}):
        sys.modules.pop("miles.utils.health_monitor", None)
        return importlib.import_module("miles.utils.health_monitor")


def _make_args(drain_grace: float = 0.0):
    return SimpleNamespace(
        rollout_health_check_interval=1.0,
        rollout_health_check_timeout=1.0,
        rollout_health_check_first_wait=0.0,
        rollout_health_kill_drain_grace_seconds=drain_grace,
    )


def test_kill_engine_disables_all_targets_once_before_shutdown():
    calls: list[str] = []
    health_monitor = _load_health_monitor_with_fake_ray(calls)
    engines = [_FakeEngine("e0", calls), _FakeEngine("e1", calls)]
    server_group = SimpleNamespace(nodes_per_engine=2, all_engines=list(engines))
    monitor = health_monitor.RolloutHealthMonitor(server_group, _make_args(drain_grace=0.0))

    monitor._kill_engine(rollout_engine_id=0)

    assert calls == [
        "e0:disable",
        "ray.get:2.0",
        "e1:disable",
        "ray.get:2.0",
        "e0:shutdown",
        "ray.get:10.0",
        "e0:ray.kill",
        "e1:shutdown",
        "ray.get:10.0",
        "e1:ray.kill",
    ]
    assert server_group.all_engines == [None, None]


def test_kill_engine_uses_one_shared_drain_grace():
    calls: list[str] = []
    health_monitor = _load_health_monitor_with_fake_ray(calls)
    engines = [_FakeEngine("e0", calls), _FakeEngine("e1", calls)]
    server_group = SimpleNamespace(nodes_per_engine=2, all_engines=list(engines))
    monitor = health_monitor.RolloutHealthMonitor(server_group, _make_args(drain_grace=2.0))

    with mock.patch.object(health_monitor.time, "sleep", side_effect=lambda seconds: calls.append(f"sleep:{seconds}")):
        monitor._kill_engine(rollout_engine_id=0)

    assert calls.count("sleep:2.0") == 1
    assert calls.index("sleep:2.0") > calls.index("e1:disable")
    assert calls.index("sleep:2.0") < calls.index("e0:shutdown")


def test_drain_window_preserves_accounting_until_shutdown():
    calls: list[str] = []
    health_monitor = _load_health_monitor_with_fake_ray(calls)
    worker_url = "http://w1:8000"
    enabled_workers = {worker_url}
    worker_request_counts = {worker_url: 1}

    def disable():
        enabled_workers.discard(worker_url)

    def finish_in_flight_during_grace(seconds):
        calls.append(f"sleep:{seconds}")
        assert worker_url in worker_request_counts
        worker_request_counts[worker_url] -= 1

    def shutdown():
        worker_request_counts.pop(worker_url, None)

    engine = _FakeEngine("e0", calls, on_disable=disable, on_shutdown=shutdown)
    server_group = SimpleNamespace(nodes_per_engine=1, all_engines=[engine])
    monitor = health_monitor.RolloutHealthMonitor(server_group, _make_args(drain_grace=2.0))

    with mock.patch.object(health_monitor.time, "sleep", side_effect=finish_in_flight_during_grace):
        monitor._kill_engine(rollout_engine_id=0)

    assert worker_url not in enabled_workers
    assert worker_url not in worker_request_counts
    assert calls == [
        "e0:disable",
        "ray.get:2.0",
        "sleep:2.0",
        "e0:shutdown",
        "ray.get:10.0",
        "e0:ray.kill",
    ]


def test_disable_failure_still_hard_kills_engine():
    calls: list[str] = []
    health_monitor = _load_health_monitor_with_fake_ray(calls)

    def disable_failure():
        raise RuntimeError("actor is not responding")

    engine = _FakeEngine("e0", calls, on_disable=disable_failure)
    server_group = SimpleNamespace(nodes_per_engine=1, all_engines=[engine])
    monitor = health_monitor.RolloutHealthMonitor(server_group, _make_args(drain_grace=0.0))

    monitor._kill_engine(rollout_engine_id=0)

    assert calls == [
        "e0:disable",
        "e0:shutdown",
        "ray.get:10.0",
        "e0:ray.kill",
    ]
    assert server_group.all_engines == [None]


def test_kill_engine_does_not_sleep_when_all_targets_are_none():
    calls: list[str] = []
    health_monitor = _load_health_monitor_with_fake_ray(calls)
    server_group = SimpleNamespace(nodes_per_engine=2, all_engines=[None, None])
    monitor = health_monitor.RolloutHealthMonitor(server_group, _make_args(drain_grace=2.0))

    with mock.patch.object(health_monitor.time, "sleep", side_effect=lambda seconds: calls.append(f"sleep:{seconds}")):
        monitor._kill_engine(rollout_engine_id=0)

    assert calls == []
    assert server_group.all_engines == [None, None]
