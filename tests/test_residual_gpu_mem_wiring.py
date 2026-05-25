from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlix_drivers_forward_residual_gpu_mem_env_var() -> None:
    for relpath in (
        "examples/rlix/run_miles_rlix.py",
        "examples/rlix/run_miles_dual.py",
    ):
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert (
            '"MILES_MAX_RESIDUAL_GPU_MEM_GB"' in source
        ), f"{relpath} must forward residual threshold env into runtime_env"


def test_shrink_logs_observed_sglang_residual_allocation() -> None:
    source = (REPO_ROOT / "miles" / "ray" / "rollout.py").read_text(
        encoding="utf-8"
    )
    assert "observed_vram_gbs = ray.get" in source
    assert "post-sleep SGLang residual allocation" in source
