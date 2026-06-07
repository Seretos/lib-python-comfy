"""Flow runner — orchestrates prompt submission, polling, and result retrieval.

Wraps :class:`~lib_python_comfy.client.ComfyClient` (synchronous) with
``asyncio.to_thread`` so every blocking call is safe to await.
:class:`~lib_python_comfy.serialization.SerializationGuard` ensures only one
prompt is submitted at a time from any given :class:`FlowRunner` instance.

Typical usage::

    client = ComfyClient("http://127.0.0.1:8188")
    guard  = SerializationGuard()
    runner = FlowRunner(client, guard)

    result = await runner.run(api_prompt, timeout=120.0)
    if result.state == JobState.COMPLETED:
        print(result.outputs)
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
from typing import Any

from lib_python_comfy.client import ComfyClient
from lib_python_comfy.serialization import SerializationGuard


class JobState(enum.Enum):
    """Possible states of a ComfyUI prompt job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Result returned by :meth:`FlowRunner.run`.

    Attributes
    ----------
    prompt_id:
        The identifier assigned by ComfyUI when the prompt was enqueued.
    state:
        One of :attr:`JobState.COMPLETED`, :attr:`JobState.ERROR`, or
        :attr:`JobState.RUNNING` (when the timeout was reached before the
        job finished).
    outputs:
        Raw outputs dict from the history entry; ``{}`` when the job did
        not reach a terminal state before the timeout.
    history:
        Full history entry dict; ``{}`` when not yet in history.
    error:
        Human-readable error description when *state* is
        :attr:`JobState.ERROR`; ``None`` otherwise.
    """

    prompt_id: str
    state: JobState
    outputs: dict
    history: dict
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class JobStatus:
    """Status returned by :meth:`FlowRunner.get_job`.

    Attributes
    ----------
    prompt_id:
        The prompt identifier that was queried.
    state:
        One of the five :class:`JobState` values.
    history:
        Full history entry dict; ``{}`` when the job is not in history.
    error:
        Human-readable error description when *state* is
        :attr:`JobState.ERROR`; ``None`` otherwise.
    """

    prompt_id: str
    state: JobState
    history: dict
    error: str | None = None


class FlowRunner:
    """Orchestrates prompt submission, polling, and result retrieval.

    Parameters
    ----------
    client:
        Synchronous :class:`~lib_python_comfy.client.ComfyClient` instance.
        All blocking calls are wrapped with ``asyncio.to_thread``.
    guard:
        :class:`~lib_python_comfy.serialization.SerializationGuard` that
        serializes concurrent prompt submissions.  A single guard may be
        shared across multiple :class:`FlowRunner` instances if desired.
    poll_interval:
        Seconds to wait between history-poll iterations.  Defaults to
        ``1.0``; set to ``0.0`` in tests for speed.
    """

    def __init__(
        self,
        client: ComfyClient,
        guard: SerializationGuard,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        self._client = client
        self._guard = guard
        self._poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, prompt: dict, *, timeout: float) -> RunResult:
        """Submit *prompt* and poll until completion or *timeout* expires.

        The guard is acquired only for the submission step; polling runs
        outside the guard so other callers can submit while we wait.

        Parameters
        ----------
        prompt:
            API-format workflow dict (as produced by
            :func:`~lib_python_comfy.graph.to_api`).
        timeout:
            Maximum number of seconds to wait for the job to reach a
            terminal state.  When the deadline is reached and the job is
            still in progress, a :attr:`JobState.RUNNING` result is
            returned with empty *outputs* and *history*.

        Returns
        -------
        RunResult
            *state* is :attr:`JobState.COMPLETED` on success,
            :attr:`JobState.ERROR` when ComfyUI reports a failure, or
            :attr:`JobState.RUNNING` on timeout.

        Raises
        ------
        ComfyConnectionError
            Propagated unhandled from any failing HTTP call.
        asyncio.CancelledError
            Propagated if the task is cancelled while polling.
        """
        async with self._guard:
            prompt_id: str = await asyncio.to_thread(
                self._client.queue_prompt, prompt
            )

        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            raw: dict[str, Any] = await asyncio.to_thread(
                self._client.get_history, prompt_id
            )
            entry = raw.get(prompt_id)
            if entry is not None:
                status_str = entry.get("status", {}).get("status_str", "")
                if status_str == "success":
                    return RunResult(
                        prompt_id=prompt_id,
                        state=JobState.COMPLETED,
                        outputs=entry.get("outputs", {}),
                        history=entry,
                    )
                return RunResult(
                    prompt_id=prompt_id,
                    state=JobState.ERROR,
                    outputs={},
                    history=entry,
                    error=str(entry.get("status")),
                )

            if asyncio.get_event_loop().time() >= deadline:
                return RunResult(
                    prompt_id=prompt_id,
                    state=JobState.RUNNING,
                    outputs={},
                    history={},
                )

            await asyncio.sleep(self._poll_interval)

    async def get_job(self, prompt_id: str) -> JobStatus:
        """Classify *prompt_id* across five states.

        Checks the live queue first (running → queued), then falls back to
        history (completed → error → not_found).

        Parameters
        ----------
        prompt_id:
            The identifier assigned by ComfyUI when the prompt was enqueued.

        Returns
        -------
        JobStatus
            *state* reflects the current position of the prompt:

            * :attr:`JobState.RUNNING`   — currently executing.
            * :attr:`JobState.QUEUED`    — waiting in the pending queue.
            * :attr:`JobState.COMPLETED` — finished successfully.
            * :attr:`JobState.ERROR`     — finished with an error.
            * :attr:`JobState.NOT_FOUND` — not in queue or history.

        Raises
        ------
        ComfyConnectionError
            Propagated unhandled from any failing HTTP call.
        """
        queue: dict[str, Any] = await asyncio.to_thread(self._client.get_queue)

        # Queue entry shape: [number, prompt_id, prompt_dict, extra_data, outputs_to_execute]
        for entry in queue.get("queue_running", []):
            if entry[1] == prompt_id:
                return JobStatus(
                    prompt_id=prompt_id, state=JobState.RUNNING, history={}
                )

        for entry in queue.get("queue_pending", []):
            if entry[1] == prompt_id:
                return JobStatus(
                    prompt_id=prompt_id, state=JobState.QUEUED, history={}
                )

        raw: dict[str, Any] = await asyncio.to_thread(
            self._client.get_history, prompt_id
        )
        entry = raw.get(prompt_id)
        if entry is None:
            return JobStatus(
                prompt_id=prompt_id, state=JobState.NOT_FOUND, history={}
            )

        status_str = entry.get("status", {}).get("status_str", "")
        if status_str == "success":
            return JobStatus(
                prompt_id=prompt_id, state=JobState.COMPLETED, history=entry
            )
        return JobStatus(
            prompt_id=prompt_id,
            state=JobState.ERROR,
            history=entry,
            error=str(entry.get("status")),
        )
