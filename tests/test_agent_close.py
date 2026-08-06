import asyncio
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _OfflineProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature
        self.calls += 1
        return LLMResponse(content="offline")

    def get_default_model(self) -> str:
        return "offline"


def _loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(), provider=_OfflineProvider(), workspace=tmp_path,
        deepthink_enabled=False,
    )


@pytest.mark.asyncio
async def test_agent_close_propagates_consolidation_finalizer_failure(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    started = asyncio.Event()

    async def failing_worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("finalizer failed")

    worker = asyncio.create_task(failing_worker())
    loop._consolidation_tasks["fixture"] = worker
    await started.wait()
    with pytest.raises(RuntimeError, match="finalizer failed"):
        await loop.aclose()
    assert worker.done()
    assert loop._consolidation_tasks == {}


@pytest.mark.asyncio
async def test_agent_close_preserves_caller_cancellation_and_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def failing_subagent_close() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        raise RuntimeError("subagent cleanup failed")

    monkeypatch.setattr(loop.subagents, "aclose", failing_subagent_close)
    caller = asyncio.create_task(loop.aclose())
    await cleanup_started.wait()
    caller.cancel()
    caller.cancel()
    release_cleanup.set()
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await caller
    leaves = list(exc_info.value.exceptions)
    assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
    assert any(isinstance(error, RuntimeError) for error in leaves)


@pytest.mark.asyncio
async def test_agent_close_is_terminal_for_public_entrypoints(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await loop.process_direct("after close")
    with pytest.raises(RuntimeError, match="closed"):
        await loop.run()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["run", "direct"])
async def test_agent_close_cancels_entry_blocked_before_scope_lease(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    loop = _loop(tmp_path)
    provider = loop.provider
    assert isinstance(provider, _OfflineProvider)
    await loop._mcp_lifecycle_lock.acquire()
    operation = asyncio.create_task(
        loop.run() if entrypoint == "run"
        else loop.process_direct("must not execute after close")
    )
    try:
        await asyncio.sleep(0)
        assert operation in loop._agent_active_entries
        await loop.aclose()
    finally:
        loop._mcp_lifecycle_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert not loop._running
    assert loop._agent_active_entries == {}
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_agent_close_waits_for_in_flight_provider_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    provider = loop.provider
    assert isinstance(provider, _OfflineProvider)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def blocking_chat(*args: Any, **kwargs: Any) -> LLMResponse:
        del args, kwargs
        provider.calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            raise
        return LLMResponse(content="must not return after close")

    monkeypatch.setattr(provider, "chat", blocking_chat)
    request = asyncio.create_task(loop.process_direct("in flight"))
    await started.wait()
    closing = asyncio.create_task(loop.aclose())
    await cancelled.wait()
    await asyncio.sleep(0)
    assert not closing.done()
    assert not request.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    await closing
    assert loop._agent_active_entries == {}
    assert provider.calls == 1
