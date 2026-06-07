# TMS hook-mode arch guard + RLix offload default

**Date:** 2026-06-06
**Branch:** `zhenyu/tms-arch-guard-rlix-offload` (off `zhenyu/m11-mvp-test`)
**Test box:** vast.ai `ssh5.vast.ai:16201` — 4× NVIDIA L4 (Ada, **sm_89**), torch 2.11.0+cu129, **CUDA 12.9**

This note is the changelog **and** the test-results record for the change. It is
append-only — later updates add new sections rather than editing existing ones.

---

## 1. Why

Two adjacent footguns in the SGLang sleep/wake (F1) path, both of which fail
*silently or unhelpfully*:

1. **`torch_memory_saver` (tms) `preload` hook segfaults on Blackwell.**
   tms' default hook mode, `preload`, is an `LD_PRELOAD` libc-malloc interposer.
   On Blackwell-class GPUs (compute capability major ≥ 10 — RTX 50xx `sm_120`,
   B100/B200 `sm_100`) under CUDA 12.9 it crashes on the first allocation
   (`build_cpu_bucket_cache`) with a **raw SIGSEGV and no Python traceback**. The
   only fix is `MILES_TMS_HOOK_MODE=torch`, but if you forget the export tms
   silently defaults back to `preload` and you get the segfault again.

2. **RLix mode needs `--offload-rollout` but doesn't enforce it.**
   `enable_memory_saver` (the flag that lets SGLang actually return VRAM to the
   OS on `release_memory_occupation`) is gated by `args.offload_rollout`
   (`backends/sglang_utils/sglang_engine.py:1122`). Under RLix time-sharing this
   is mandatory — without it `release_memory_occupation` is a silent no-op and
   the first `shrink_engines` OOMs (the M11.1 attempt-5 bug). But nothing forced
   it on; the operator had to remember the CLI flag.

> Capability is good, but *failing loudly* matters just as much. Both changes
> turn a silent/tracebackless failure into either a clear error or correct
> default behavior.

---

## 2. What changed

| File | Change |
|---|---|
| `miles/backends/megatron_utils/tms_utils.py` | **New.** `resolve_tms_hook_mode()` + `assert_tms_hook_mode_matches_arch()`. torch-only module so the guard is unit-testable without importing the Megatron actor stack. |
| `miles/backends/megatron_utils/actor.py` | Import + call `assert_tms_hook_mode_matches_arch(mode)` right before the tms `hook_mode` is applied (inside `if args.offload_train:`). |
| `miles/utils/rlix_validation.py` | **New helper** `apply_rlix_offload_defaults(args)` — forces `offload_rollout=True` under RLix. Idempotent; leaves `offload_train` untouched. |
| `miles/utils/arguments.py` | Call `apply_rlix_offload_defaults(args)` in `miles_validate_args`, right after `offload_rollout` defaults to `False` (single normalization point → all downstream `if args.offload_rollout` checks stay consistent). |
| `tests/fast/backends/test_tms_utils.py` | **New.** 19 cases for the arch guard. |
| `tests/fast/utils/test_rlix_offload_defaults.py` | **New.** 5 cases for the RLix offload default. |

### Guard semantics (deliberate choices)
- Checks the **resolved** mode (`unset → preload`), so the "forgot the export"
  case is caught, not just an explicit `preload`.
- Fires only on **major ≥ 10**, so pre-Blackwell (V100/A100/Ada/Hopper) keeps
  `preload` — which is the *more complete* release path and tms' default.
- **Not** hardcoded to torch: torch mode has narrower catchment (only PyTorch
  allocations). The right layer to pin torch is the deployment, not the library.
- Escape hatch `MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL=1` for when a fixed
  tms/CUDA build is confirmed (mirrors the repo's `MILES_SKIP_*` convention).
- Raises `RuntimeError`, not bare `assert`, so it survives `python -O` (matches
  the `rlix_validation.py` convention).
- RLix coupling forces **only** `offload_rollout`, not `offload_train` (separate
  train-side knob with its own cost/benefit).

---

## 3. Test results

Run on the vast L4 box, `PYTHONPATH=/root/miles_pr`:

```
$ python -m pytest tests/fast/utils/test_rlix_offload_defaults.py \
                   tests/fast/backends/test_tms_utils.py -q
======================= 24 passed, 29 warnings in 8.42s ========================
```

- `test_tms_utils.py` — 19 passed: resolver table (5), preload-on-Blackwell
  raises for sm_100 & sm_120 across unset/preload/bogus (6), pre-Blackwell
  no-raise for sm_70/80/89/90 (4), torch-mode-always-safe (2), escape hatch (1),
  no-CUDA no-op (1).
- `test_rlix_offload_defaults.py` — 5 passed: forces on under RLix, no-op when
  not RLix / other control plane, idempotent when already on, leaves
  `offload_train` alone.

### Live (non-mocked) hardware check on the real L4 (sm_89)
```
actor import OK after refactor
real device: NVIDIA L4 (8, 9)
live preload on L4: no raise (correct)   # major 8 < 10 → guard stays silent
live unset on L4:   no raise (correct)
live torch on L4:   no raise (correct)
resolve(None)= preload
```
Confirms (a) `actor.py` still imports after the refactor, and (b) the guard does
not false-positive on a real pre-Blackwell GPU. The Blackwell *raise* path is
covered by the mocked unit tests (no Blackwell GPU on this box).

---

## 4. Codex sign-off

Codex review verdict: **APPROVE** — no CRITICAL / HIGH / MEDIUM findings. One LOW
note: `apply_rlix_offload_defaults` mutates `args` in place. This is intentional
— it matches the pervasive in-place normalization style of `miles_validate_args`
and is documented in the helper's docstring.

---

## 5. Follow-ups (not in this change)
- Consider auto-detecting hook mode (default `torch` when CC ≥ 10 / CUDA ≥ 12.9,
  else `preload`) with the env var as an override — would remove the "forgot the
  export" footgun entirely instead of erroring on it.
- Decide whether RLix partial-overlap also needs `offload_train` forced on (left
  untouched here on purpose).
