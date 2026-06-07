"""Single-flight serialization guard for async contexts.

Provides :class:`SerializationGuard`, an async context manager that ensures
at most one caller executes its guarded block at a time. This is useful when
multiple coroutines might concurrently attempt an operation that must not
overlap (e.g. submitting a prompt to ComfyUI while another submission is
still in flight).

Usage::

    guard = SerializationGuard()

    async def submit():
        async with guard:
            await do_exclusive_work()

Pure ``asyncio`` stdlib — no third-party dependencies.
"""
from __future__ import annotations

import asyncio


class SerializationGuard:
    """Async context manager that serializes concurrent callers.

    Wraps :class:`asyncio.Lock` so that only one coroutine at a time can
    execute the body of an ``async with guard:`` block. Any additional
    callers are suspended until the current holder exits the block.

    Exceptions raised inside the guarded block are **not** suppressed; they
    propagate normally to the caller. The lock is always released on exit,
    whether the block completed successfully or raised an exception, so the
    guard cannot deadlock.

    Example::

        guard = SerializationGuard()

        async with guard:
            await some_exclusive_operation()
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "SerializationGuard":
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._lock.release()
        return False  # do not suppress exceptions
