"""Tests for lib_python_comfy.serialization — SerializationGuard."""
import asyncio

import pytest

from lib_python_comfy import SerializationGuard


async def test_two_concurrent_callers_are_strictly_sequential():
    """Two coroutines entering the guard concurrently must run sequentially.

    Each coroutine appends "enter", yields, then appends "exit" — all inside
    the guard. When gathered, the resulting log must be strictly sequential
    (enter/exit/enter/exit), not interleaved (enter/enter/exit/exit).
    """
    guard = SerializationGuard()
    log: list[str] = []

    async def worker():
        async with guard:
            log.append("enter")
            await asyncio.sleep(0)  # yield to event loop; other coroutine may run
            log.append("exit")

    await asyncio.gather(worker(), worker())

    assert log == ["enter", "exit", "enter", "exit"], (
        f"Expected strictly sequential execution, got: {log}"
    )


async def test_exception_in_guard_releases_lock():
    """An exception raised inside the guard must not leave the lock acquired.

    After a guarded block raises, a subsequent acquire must succeed
    immediately (no deadlock). We guard against hangs with asyncio.wait_for.
    """
    guard = SerializationGuard()

    with pytest.raises(RuntimeError, match="boom"):
        async with guard:
            raise RuntimeError("boom")

    # Lock must be released; this second acquire must not hang.
    async def acquire_after_error():
        async with guard:
            pass  # just needs to succeed

    await asyncio.wait_for(acquire_after_error(), timeout=1.0)


async def test_guard_does_not_suppress_exceptions():
    """Exceptions raised inside the guard must propagate to the caller."""
    guard = SerializationGuard()

    with pytest.raises(ValueError, match="test error"):
        async with guard:
            raise ValueError("test error")


async def test_sequential_calls_succeed():
    """Baseline: sequential uses of the guard complete normally."""
    guard = SerializationGuard()

    results: list[int] = []

    async with guard:
        results.append(1)

    async with guard:
        results.append(2)

    assert results == [1, 2]


def test_serialization_guard_is_exported():
    """SerializationGuard must be importable from the public API."""
    from lib_python_comfy import SerializationGuard as SG  # noqa: F401

    assert SG is SerializationGuard
