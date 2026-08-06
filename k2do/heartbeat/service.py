"""Heartbeat service - periodic agent wake-up to check for tasks."""

import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

# Default interval: 30 minutes
DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60

# The prompt sent to agent during heartbeat
HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
If nothing needs attention, reply with just: HEARTBEAT_OK"""

# Token that indicates "nothing to do"
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content."""
    if not content:
        return True
    
    # Lines to skip: empty, headers, HTML comments, empty checkboxes
    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}
    
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--") or line in skip_patterns:
            continue
        return False  # Found actionable content
    
    return True


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.
    
    The agent reads HEARTBEAT.md from the workspace and executes any
    tasks listed there. If nothing needs attention, it replies HEARTBEAT_OK.
    """
    
    def __init__(
        self,
        workspace: Path,
        on_heartbeat: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
    ):
        self.workspace = workspace
        self.on_heartbeat = on_heartbeat
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[BaseException | None] | None = None
        self._stop_generation = 0
    
    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"
    
    def _read_heartbeat_file(self) -> str | None:
        """Read HEARTBEAT.md content."""
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text()
            except Exception:
                return None
        return None
    
    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return

        if self._running and self._task is not None and not self._task.done():
            logger.debug("Heartbeat already running")
            return

        stop_generation = self._stop_generation
        close_task = self._ensure_close_task()
        if close_task is not None:
            await self._await_close_task(close_task)
            if self._stop_generation != stop_generation:
                return
            if self._running and self._task is not None and not self._task.done():
                return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._cancel_requested_task = None
        logger.info(f"Heartbeat started (every {self.interval_s}s)")
    
    def stop(self) -> None:
        """Request heartbeat shutdown without discarding the task handle."""
        self._stop_generation += 1
        self._running = False
        self._request_task_cancel(self._task)

    async def aclose(self) -> None:
        """Cancel and await the heartbeat worker before returning."""
        self.stop()
        close_task = self._ensure_close_task()
        if close_task is not None:
            await self._await_close_task(close_task)

    def _request_task_cancel(self, task: asyncio.Task[None] | None) -> None:
        """Issue at most one service-owned cancellation for a worker."""
        if (
            task is not None
            and not task.done()
            and task.cancelling() == 0
            and self._cancel_requested_task is not task
        ):
            self._cancel_requested_task = task
            task.cancel()

    def _ensure_close_task(self) -> asyncio.Task[BaseException | None] | None:
        """Return the singleton drain task for the current worker generation."""
        close_task = self._close_task
        if close_task is not None:
            return close_task
        worker = self._task
        if worker is None:
            return None
        self._running = False
        close_task = asyncio.create_task(self._drain_worker(worker))
        self._close_task = close_task
        return close_task

    async def _drain_worker(self, worker: asyncio.Task[None]) -> BaseException | None:
        self._request_task_cancel(worker)
        expected_cancellation = self._cancel_requested_task is worker
        close_error: BaseException | None = None
        try:
            await worker
        except asyncio.CancelledError as exc:
            if not expected_cancellation:
                close_error = exc
        except BaseException as exc:
            close_error = exc
        finally:
            if self._task is worker:
                self._task = None
            if self._cancel_requested_task is worker:
                self._cancel_requested_task = None
        return close_error

    async def _await_close_task(
        self,
        close_task: asyncio.Task[BaseException | None],
    ) -> None:
        """Shield the shared drain and replay this caller's cancellation last."""
        caller_cancellation: asyncio.CancelledError | None = None
        current = asyncio.current_task()
        while not close_task.done():
            cancelling_before = current.cancelling() if current is not None else 0
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                cancelling_after = current.cancelling() if current is not None else 0
                if not close_task.done() or cancelling_after > cancelling_before:
                    if caller_cancellation is None:
                        caller_cancellation = exc
                    continue
                break
            except BaseException:
                break

        try:
            close_error = close_task.result()
        except BaseException as exc:
            close_error = exc
        if self._close_task is close_task:
            self._close_task = None

        if caller_cancellation is not None and close_error is not None:
            raise BaseExceptionGroup(
                "Heartbeat shutdown failed after caller cancellation",
                (caller_cancellation, close_error),
            ) from None
        if close_error is not None:
            raise close_error
        if caller_cancellation is not None:
            raise caller_cancellation
    
    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        content = self._read_heartbeat_file()
        
        # Skip if HEARTBEAT.md is empty or doesn't exist
        if _is_heartbeat_empty(content):
            logger.debug("Heartbeat: no tasks (HEARTBEAT.md empty)")
            return
        
        logger.info("Heartbeat: checking for tasks...")
        
        if self.on_heartbeat:
            try:
                response = await self.on_heartbeat(HEARTBEAT_PROMPT)
                
                # Check if agent said "nothing to do"
                if HEARTBEAT_OK_TOKEN.replace("_", "") in response.upper().replace("_", ""):
                    logger.info("Heartbeat: OK (no action needed)")
                else:
                    logger.info("Heartbeat: completed task")
                    
            except Exception as e:
                logger.error(f"Heartbeat execution failed: {e}")
    
    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        if self.on_heartbeat:
            return await self.on_heartbeat(HEARTBEAT_PROMPT)
        return None
