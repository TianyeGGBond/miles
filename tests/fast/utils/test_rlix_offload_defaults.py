"""Unit tests for RLix-mode offload_rollout coercion.

Imports only ``rlix_validation`` (stdlib-only), so these run anywhere with no
torch / GPU dependency.
"""

from types import SimpleNamespace

from miles.utils.rlix_validation import apply_rlix_offload_defaults


def test_forces_offload_rollout_on_under_rlix(monkeypatch):
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    args = SimpleNamespace(offload_rollout=False)
    apply_rlix_offload_defaults(args)
    assert args.offload_rollout is True


def test_no_change_when_not_rlix(monkeypatch):
    monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
    args = SimpleNamespace(offload_rollout=False)
    apply_rlix_offload_defaults(args)
    assert args.offload_rollout is False


def test_no_change_when_other_control_plane(monkeypatch):
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "standalone")
    args = SimpleNamespace(offload_rollout=False)
    apply_rlix_offload_defaults(args)
    assert args.offload_rollout is False


def test_idempotent_when_already_on(monkeypatch):
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    args = SimpleNamespace(offload_rollout=True)
    apply_rlix_offload_defaults(args)
    assert args.offload_rollout is True


def test_does_not_touch_offload_train(monkeypatch):
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    args = SimpleNamespace(offload_rollout=False, offload_train=False)
    apply_rlix_offload_defaults(args)
    assert args.offload_train is False  # train-side knob intentionally left alone
