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


def test_shrink_gates_on_process_resident_gpu_memory() -> None:
    source = (REPO_ROOT / "miles" / "ray" / "rollout.py").read_text(
        encoding="utf-8"
    )
    # Hard gate uses per-process resident GPU memory, not /server_info accounting.
    assert "assert_post_sleep_process_vram_below_threshold" in source
    assert "post-sleep process-resident GPU residual" in source
