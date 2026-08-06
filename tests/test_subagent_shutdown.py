import asyncio
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.subagent import SubagentManager
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _DelayedCancellationProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key=None, api_base=None)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = messages, tools, model, max_tokens, temperature
        self.calls += 1
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
            raise

    def get_default_model(self) -> str:
        return "fixture-model"


class _FailingProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = messages, tools, model, max_tokens, temperature
        raise RuntimeError("PRIVATE-PROVIDER-CANARY")

    def get_default_model(self) -> str:
        return "fixture-model"


def _manager(
    tmp_path: Path,
    provider: _DelayedCancellationProvider,
    bus: MessageBus,
) -> SubagentManager:
    return SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
        model="fixture-model",
    )


@pytest.mark.asyncio
async def test_aclose_cancels_and_awaits_without_error_announcement(
    tmp_path: Path,
) -> None:
    provider = _DelayedCancellationProvider()
    bus = MessageBus()
    manager = _manager(tmp_path, provider, bus)

    await manager.spawn("wait for shutdown", label="fixture")
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    workers = tuple(manager._running_tasks.values())
    assert len(workers) == 1

    close_call = asyncio.create_task(manager.aclose())
    await asyncio.wait_for(provider.cancelled.wait(), timeout=1)
    assert not close_call.done()
    assert manager.get_running_count() == 1

    provider.release.set()
    await asyncio.wait_for(close_call, timeout=1)

    assert workers[0].cancelled()
    assert manager.get_running_count() == 0
    assert bus.inbound_size == 0

    drain = manager._close_task
    await manager.aclose()
    assert manager._close_task is drain


@pytest.mark.asyncio
async def test_aclose_preserves_caller_cancellation_during_worker_teardown(
    tmp_path: Path,
) -> None:
    provider = _DelayedCancellationProvider()
    bus = MessageBus()
    manager = _manager(tmp_path, provider, bus)

    await manager.spawn("wait for shutdown", label="fixture")
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    close_call = asyncio.create_task(manager.aclose())
    await asyncio.wait_for(provider.cancelled.wait(), timeout=1)
    close_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_call, timeout=1)

    # The shielded drain remains owned by the manager until delayed worker
    # teardown finishes; cancellation of one caller does not orphan it.
    assert manager._close_task is not None
    assert not manager._close_task.done()
    assert manager.get_running_count() == 1

    provider.release.set()
    await asyncio.wait_for(manager.aclose(), timeout=1)
    assert manager.get_running_count() == 0
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_concurrent_aclose_starts_one_drain_and_rejects_new_tasks(
    tmp_path: Path,
) -> None:
    provider = _DelayedCancellationProvider()
    manager = _manager(tmp_path, provider, MessageBus())
    drain_entered = asyncio.Event()
    release_drain = asyncio.Event()
    drain_calls = 0
    original_drain = manager._drain_running_tasks

    async def counted_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        drain_entered.set()
        await release_drain.wait()
        await original_drain()

    manager._drain_running_tasks = counted_drain  # type: ignore[method-assign]

    first = asyncio.create_task(manager.aclose())
    await asyncio.wait_for(drain_entered.wait(), timeout=1)
    second = asyncio.create_task(manager.aclose())

    with pytest.raises(RuntimeError, match="manager is closed"):
        await manager.spawn("must not start")

    release_drain.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    shutdown_task = manager._close_task
    await manager.aclose()

    assert drain_calls == 1
    assert manager._close_task is shutdown_task
    assert manager.get_running_count() == 0
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_worker_failure_announcement_never_echoes_provider_details(
    tmp_path: Path,
) -> None:
    bus = MessageBus()
    manager = SubagentManager(
        provider=_FailingProvider(),
        workspace=tmp_path,
        bus=bus,
        model="fixture-model",
    )

    await manager.spawn("safe fixture", label="fixture")
    message = await asyncio.wait_for(bus.consume_inbound(), timeout=2)
    assert "PRIVATE-PROVIDER-CANARY" not in message.content
    assert "failed before producing a safe result" in message.content
    await manager.aclose()
