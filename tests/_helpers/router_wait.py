"""F77 — test-side ``asyncio.wait_for`` helpers for Gate 4 (f) sub-tests.

DEV-ONLY-MVP scaffolding. Production ``MilesRouter._use_url_async``
stays unbounded per scope F39 (C20 MVP); the bound is enforced by
the test harness so a buggy ``notify_all`` path causes CI to fail
fast at the await boundary instead of silent CI hang.

Two wrappers, matching the cozy-plan iter-29 row:
- ``await_with_positive_timeout(coro)``: 60-second bound for the
  positive sub-test (an enabled worker should be admitted promptly).
- ``await_with_counter_timeout(coro)``: 30-second bound for the
  counter sub-test (no admitted worker → suspend should outlive
  the wait, then notify_all should wake).

Not a Gate acceptance criterion; test logic uses these wrappers,
production ``_use_url_async`` does NOT.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

_T = TypeVar("_T")


async def await_with_positive_timeout(coro: Awaitable[_T]) -> _T:
    """Bound the positive Gate 4 (f) sub-test at 60 seconds.

    Used to wrap ``router._use_url_async()`` in tests where an
    admitted worker is expected to be picked up promptly. Failure
    means either the suspend predicate is wrong or the
    ``notify_all`` from ``add_worker`` / ``enable_worker`` did not
    fire.
    """
    return await asyncio.wait_for(coro, timeout=60.0)


async def await_with_counter_timeout(coro: Awaitable[_T]) -> _T:
    """Bound the counter Gate 4 (f) sub-test at 30 seconds.

    Used to wrap an admit-after-suspend scenario where the test
    enables a worker AFTER awaiting ``_use_url_async``; the wait
    must wake within 30 s of the matching ``notify_all``.
    """
    return await asyncio.wait_for(coro, timeout=30.0)


__all__ = [
    "await_with_positive_timeout",
    "await_with_counter_timeout",
]
