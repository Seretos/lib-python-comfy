"""Tests for lib_python_comfy.runner — FlowRunner, JobState, JobStatus, RunResult."""
import asyncio
from unittest.mock import MagicMock

import pytest

from lib_python_comfy.client import ComfyConnectionError
from lib_python_comfy.runner import FlowRunner, JobState, JobStatus, RunResult
from lib_python_comfy.serialization import SerializationGuard


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_runner(client: MagicMock, poll_interval: float = 0.0) -> FlowRunner:
    """Return a FlowRunner with a fresh SerializationGuard."""
    return FlowRunner(client, SerializationGuard(), poll_interval=poll_interval)


def _mock_client(
    *,
    prompt_id: str = "pid-1",
    history_side_effect=None,
    history_return_value=None,
    queue_return_value=None,
) -> MagicMock:
    """Build a MagicMock ComfyClient with sensible defaults."""
    client = MagicMock()
    client.queue_prompt.return_value = prompt_id
    if history_side_effect is not None:
        client.get_history.side_effect = history_side_effect
    elif history_return_value is not None:
        client.get_history.return_value = history_return_value
    else:
        client.get_history.return_value = {}
    if queue_return_value is not None:
        client.get_queue.return_value = queue_return_value
    else:
        client.get_queue.return_value = {"queue_running": [], "queue_pending": []}
    return client


# ---------------------------------------------------------------------------
# run() — happy-path: job already complete on first poll
# ---------------------------------------------------------------------------


async def test_run_returns_completed_result_when_history_populates_immediately():
    """queue_prompt returns a prompt_id; get_history immediately shows success."""
    outputs = {"node1": {"images": []}}
    history_entry = {"status": {"status_str": "success"}, "outputs": outputs}
    client = _mock_client(
        prompt_id="pid-1",
        history_return_value={"pid-1": history_entry},
    )
    runner = _make_runner(client)

    result = await runner.run({"dummy": "prompt"}, timeout=5.0)

    assert isinstance(result, RunResult)
    assert result.state == JobState.COMPLETED
    assert result.prompt_id == "pid-1"
    assert result.outputs == outputs
    assert result.error is None
    assert result.history == history_entry


# ---------------------------------------------------------------------------
# run() — timeout reached before history populates
# ---------------------------------------------------------------------------


async def test_run_returns_running_result_on_timeout():
    """When get_history always returns {} and timeout=0.0, result is RUNNING."""
    client = _mock_client(prompt_id="pid-1", history_return_value={})
    runner = _make_runner(client)

    result = await runner.run({"dummy": "prompt"}, timeout=0.0)

    assert result.state == JobState.RUNNING
    assert result.prompt_id == "pid-1"
    assert result.outputs == {}
    assert result.history == {}


# ---------------------------------------------------------------------------
# run() — history entry present but reports error
# ---------------------------------------------------------------------------


async def test_run_returns_error_result_when_history_contains_error():
    """When history status_str is 'error', RunResult.state must be ERROR."""
    error_status = {
        "status_str": "error",
        "messages": [["execution_error", {}]],
    }
    history_entry = {"status": error_status, "outputs": {}}
    client = _mock_client(
        prompt_id="pid-1",
        history_return_value={"pid-1": history_entry},
    )
    runner = _make_runner(client)

    result = await runner.run({"dummy": "prompt"}, timeout=5.0)

    assert result.state == JobState.ERROR
    assert result.prompt_id == "pid-1"
    assert result.error is not None
    assert result.history == history_entry


# ---------------------------------------------------------------------------
# run() — cancellation mid-poll
# ---------------------------------------------------------------------------


async def test_run_cancel_mid_poll():
    """Cancelling a running task propagates CancelledError; guard is released."""
    # get_history returns {} on every call so the loop keeps spinning
    call_count = 0

    def _history_called(_prompt_id: str) -> dict:
        nonlocal call_count
        call_count += 1
        return {}

    client = _mock_client(prompt_id="pid-1")
    client.get_history.side_effect = _history_called

    guard = SerializationGuard()
    runner = FlowRunner(client, guard, poll_interval=0.01)

    task = asyncio.create_task(runner.run({"dummy": "prompt"}, timeout=60.0))

    # Wait until polling has demonstrably started (get_history called at least
    # once) before cancelling. Use real sleeps with a generous deadline: the
    # to_thread worker needs wall-clock time to run, and asyncio.sleep(0) yields
    # without advancing time, which races the thread pool on loaded CI hosts.
    deadline = asyncio.get_event_loop().time() + 5.0
    while call_count < 1 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)

    assert call_count >= 1, "get_history should have been called before cancel"

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Guard must be released — another run must not hang.
    client2 = _mock_client(
        prompt_id="pid-2",
        history_return_value={"pid-2": {"status": {"status_str": "success"}, "outputs": {}}},
    )
    runner2 = FlowRunner(client2, guard, poll_interval=0.0)
    result = await asyncio.wait_for(
        runner2.run({"dummy": "prompt"}, timeout=5.0), timeout=2.0
    )
    assert result.state == JobState.COMPLETED


# ---------------------------------------------------------------------------
# run() — ComfyConnectionError propagation
# ---------------------------------------------------------------------------


async def test_run_propagates_connection_error():
    """ComfyConnectionError raised by get_history must propagate out of run()."""
    client = _mock_client(prompt_id="pid-1")
    client.get_history.side_effect = ComfyConnectionError("network failure")
    runner = _make_runner(client)

    with pytest.raises(ComfyConnectionError, match="network failure"):
        await runner.run({"dummy": "prompt"}, timeout=5.0)


# ---------------------------------------------------------------------------
# get_job() — all five states
# ---------------------------------------------------------------------------


async def test_get_job_running():
    """Prompt found in queue_running → JobState.RUNNING."""
    client = _mock_client(
        queue_return_value={
            "queue_running": [[0, "pid-1", {}, {}, []]],
            "queue_pending": [],
        }
    )
    runner = _make_runner(client)

    status = await runner.get_job("pid-1")

    assert isinstance(status, JobStatus)
    assert status.state == JobState.RUNNING
    assert status.prompt_id == "pid-1"
    assert status.history == {}
    assert status.error is None


async def test_get_job_queued():
    """Prompt found in queue_pending → JobState.QUEUED."""
    client = _mock_client(
        queue_return_value={
            "queue_running": [],
            "queue_pending": [[0, "pid-1", {}, {}, []]],
        }
    )
    runner = _make_runner(client)

    status = await runner.get_job("pid-1")

    assert status.state == JobState.QUEUED
    assert status.prompt_id == "pid-1"
    assert status.history == {}


async def test_get_job_completed():
    """Prompt in history with status_str 'success' → JobState.COMPLETED."""
    history_entry = {"status": {"status_str": "success"}, "outputs": {"n": {}}}
    client = _mock_client(
        queue_return_value={"queue_running": [], "queue_pending": []},
        history_return_value={"pid-1": history_entry},
    )
    runner = _make_runner(client)

    status = await runner.get_job("pid-1")

    assert status.state == JobState.COMPLETED
    assert status.history == history_entry
    assert status.error is None


async def test_get_job_error():
    """Prompt in history with non-success status_str → JobState.ERROR."""
    error_status = {"status_str": "error", "messages": []}
    history_entry = {"status": error_status, "outputs": {}}
    client = _mock_client(
        queue_return_value={"queue_running": [], "queue_pending": []},
        history_return_value={"pid-1": history_entry},
    )
    runner = _make_runner(client)

    status = await runner.get_job("pid-1")

    assert status.state == JobState.ERROR
    assert status.error is not None
    assert status.history == history_entry


async def test_get_job_not_found():
    """Prompt absent from queue and history → JobState.NOT_FOUND."""
    client = _mock_client(
        queue_return_value={"queue_running": [], "queue_pending": []},
        history_return_value={},
    )
    runner = _make_runner(client)

    status = await runner.get_job("pid-1")

    assert status.state == JobState.NOT_FOUND
    assert status.history == {}
    assert status.error is None


# ---------------------------------------------------------------------------
# run() — concurrent submissions are serialized
# ---------------------------------------------------------------------------


async def test_run_serializes_concurrent_submissions():
    """Two concurrent run() calls must not submit prompts simultaneously.

    We track when queue_prompt is *entered* and *exited* using a shared
    list, then verify the calls were strictly sequential (no overlap).
    """
    active_submissions = 0
    max_concurrent = 0
    submission_order: list[str] = []

    def _queue_prompt_first(prompt: dict) -> str:
        nonlocal active_submissions, max_concurrent
        active_submissions += 1
        max_concurrent = max(max_concurrent, active_submissions)
        submission_order.append("start-1")
        result = "pid-1"
        active_submissions -= 1
        return result

    def _queue_prompt_second(prompt: dict) -> str:
        nonlocal active_submissions, max_concurrent
        active_submissions += 1
        max_concurrent = max(max_concurrent, active_submissions)
        submission_order.append("start-2")
        result = "pid-2"
        active_submissions -= 1
        return result

    history_data = {
        "pid-1": {"status": {"status_str": "success"}, "outputs": {}},
        "pid-2": {"status": {"status_str": "success"}, "outputs": {}},
    }

    client = MagicMock()
    client.queue_prompt.side_effect = [
        _queue_prompt_first({}),
        _queue_prompt_second({}),
    ]
    # Use a list to simulate sequential side effects properly
    client.queue_prompt.side_effect = None
    _calls = [_queue_prompt_first, _queue_prompt_second]
    _call_iter = iter(_calls)

    def _dispatch(prompt: dict) -> str:
        return next(_call_iter)(prompt)

    client.queue_prompt.side_effect = _dispatch

    def _get_history(prompt_id: str) -> dict:
        return {prompt_id: history_data[prompt_id]} if prompt_id in history_data else {}

    client.get_history.side_effect = _get_history

    guard = SerializationGuard()
    runner = FlowRunner(client, guard, poll_interval=0.0)

    results = await asyncio.gather(
        runner.run({"prompt": "A"}, timeout=5.0),
        runner.run({"prompt": "B"}, timeout=5.0),
    )

    # Both must succeed
    assert all(r.state == JobState.COMPLETED for r in results)

    # Submissions must be strictly sequential — max concurrency never > 1
    assert max_concurrent <= 1, (
        f"queue_prompt was active in more than one thread simultaneously "
        f"(max_concurrent={max_concurrent})"
    )


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_runner_types_are_exported():
    """FlowRunner, RunResult, JobStatus, JobState must be importable from the package."""
    import lib_python_comfy as pkg
    import lib_python_comfy.runner as runner_mod

    assert pkg.FlowRunner is runner_mod.FlowRunner
    assert pkg.RunResult is runner_mod.RunResult
    assert pkg.JobStatus is runner_mod.JobStatus
    assert pkg.JobState is runner_mod.JobState
