from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import (
    _MCP_INHERITED_SCOPE_REVOKED,
    AgentLoop,
    _log_tool_call,
)
from k2do.agent.tools.base import Tool
from k2do.agent.tools.mcp import (
    MCPConnectionReport,
    MCPServerOutcome,
    mcp_public_name,
)
from k2do.agent.tools.registry import MCP_INVALID_NAME_MARKER
from k2do.bus.events import OutboundMessage
from k2do.bus.queue import MessageBus
from k2do.cli.commands import _run_gateway_cleanup
from k2do.cron.service import CronService
from k2do.cron.types import CronSchedule
from k2do.heartbeat.service import HeartbeatService
from k2do.providers.base import LLMProvider, LLMResponse


class _Provider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature
        return LLMResponse(content="fixture")

    def get_default_model(self) -> str:
        return "offline/fixture"


class _RemoteTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "fixture"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return "ok"


def _loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_Provider(),
        workspace=tmp_path,
        model="offline/fixture",
        fallback_model="offline/fixture",
        mcp_servers={"fixture": object()},
        deepthink_enabled=False,
    )


def _loop_without_mcp(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_Provider(),
        workspace=tmp_path,
        model="offline/fixture",
        fallback_model="offline/fixture",
        deepthink_enabled=False,
    )


def _register_report(loop: AgentLoop, stack: AsyncExitStack) -> MCPConnectionReport:
    tool = _RemoteTool(mcp_public_name("fixture", "probe"))
    names = loop.tools.register_many([tool])
    return MCPConnectionReport(
        outcomes=(
            MCPServerOutcome(
                position=0,
                status="connected",
                phase="ready",
                protocol_version="2025-11-25",
                tool_names=names,
            ),
        ),
        registered_tool_names=names,
        registered_tools=(tool,),
    )


@pytest.mark.asyncio
async def test_no_server_configuration_does_not_publish_or_serialize_mcp_state(
    tmp_path: Path,
) -> None:
    loop = _loop_without_mcp(tmp_path)
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        nonlocal entered
        del session_key
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=msg.content,
        )

    loop._process_message = process_message  # type: ignore[method-assign]
    first = asyncio.create_task(loop.process_direct("first"))
    second = asyncio.create_task(loop.process_direct("second"))
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    assert loop._mcp_connected is False
    assert loop._mcp_scope_lock.locked() is False

    release.set()
    assert await asyncio.gather(first, second) == ["first", "second"]


@pytest.mark.asyncio
async def test_close_unregisters_wrappers_resets_state_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    baseline = tuple(loop.tools.tool_names)
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    assert await loop._connect_mcp() is True
    assert len(loop.tools) == len(baseline) + 1
    assert loop._mcp_connected is True

    await loop.close_mcp()
    assert tuple(loop.tools.tool_names) == baseline
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert cleanup == ["closed"]

    await loop.close_mcp()
    assert cleanup == ["closed"]


@pytest.mark.asyncio
async def test_cancelled_startup_unwinds_and_the_same_loop_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    attempts = 0
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal attempts
        del servers, registry
        attempts += 1
        stack.callback(cleanup.append, f"attempt-{attempts}")
        if attempts == 1:
            raise asyncio.CancelledError
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    with pytest.raises(asyncio.CancelledError):
        await loop._connect_mcp()
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert cleanup == ["attempt-1"]

    assert await loop._connect_mcp() is True
    assert attempts == 2
    await loop.close_mcp()
    assert cleanup == ["attempt-1", "attempt-2"]


@pytest.mark.asyncio
async def test_caller_cancellation_during_startup_reaps_owner_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    startup_entered = asyncio.Event()
    block_first_startup = asyncio.Event()
    connect_tasks: list[asyncio.Task[Any] | None] = []
    cleanup_tasks: list[asyncio.Task[Any] | None] = []
    attempts = 0

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal attempts
        del servers, registry
        attempts += 1
        connect_tasks.append(asyncio.current_task())

        def record_cleanup_task() -> None:
            cleanup_tasks.append(asyncio.current_task())

        stack.callback(record_cleanup_task)
        if attempts == 1:
            startup_entered.set()
            await block_first_startup.wait()
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    caller = asyncio.create_task(loop._connect_mcp())
    await asyncio.wait_for(startup_entered.wait(), timeout=1)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=1)

    assert connect_tasks == cleanup_tasks
    assert connect_tasks[0] is not caller
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert loop._mcp_transport_task is None

    assert await loop._connect_mcp() is True
    await loop.close_mcp()
    assert attempts == 2
    assert connect_tasks == cleanup_tasks


@pytest.mark.asyncio
async def test_repeated_caller_cancel_during_startup_cleanup_does_not_interrupt_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_completed = asyncio.Event()
    connect_tasks: list[asyncio.Task[Any] | None] = []
    cleanup_tasks: list[asyncio.Task[Any] | None] = []
    attempts = 0
    resource_open = True

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal attempts, resource_open
        del servers, registry
        attempts += 1
        connect_tasks.append(asyncio.current_task())

        if attempts == 1:
            resource_open = True

            async def slow_startup_cleanup() -> None:
                nonlocal resource_open
                cleanup_tasks.append(asyncio.current_task())
                cleanup_started.set()
                await release_cleanup.wait()
                resource_open = False
                cleanup_completed.set()

            stack.push_async_callback(slow_startup_cleanup)
            raise RuntimeError("startup failed")

        def record_cleanup_task() -> None:
            cleanup_tasks.append(asyncio.current_task())

        stack.callback(record_cleanup_task)
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    caller = asyncio.create_task(loop._connect_mcp())
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    transport_owner = loop._mcp_transport_task
    assert transport_owner is not None

    caller.cancel()
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)
    assert caller.done() is False
    assert transport_owner.done() is False
    assert transport_owner.cancelling() == 0
    assert resource_open is True
    assert cleanup_completed.is_set() is False

    release_cleanup.set()
    with pytest.raises(BaseExceptionGroup) as captured:
        await asyncio.wait_for(caller, timeout=1)

    assert [type(error) for error in captured.value.exceptions] == [
        asyncio.CancelledError,
        RuntimeError,
    ]
    assert cleanup_completed.is_set()
    assert resource_open is False
    assert connect_tasks == cleanup_tasks
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert loop._mcp_transport_task is None

    assert await loop._connect_mcp() is True
    await loop.close_mcp()
    assert attempts == 2
    assert connect_tasks == cleanup_tasks


@pytest.mark.asyncio
async def test_transport_owner_cancelled_before_first_step_resets_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    real_create_task = asyncio.create_task

    def create_cancelled_task(
        coroutine: Any,
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coroutine, name=name)
        task.cancel()
        return task

    monkeypatch.setattr(asyncio, "create_task", create_cancelled_task)
    caller = asyncio.get_running_loop().create_task(loop._connect_mcp())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=1)

    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert loop._mcp_transport_task is None

    monkeypatch.setattr(asyncio, "create_task", real_create_task)
    assert await loop._connect_mcp() is True
    await loop.close_mcp()
    assert loop._mcp_transport_task is None


@pytest.mark.asyncio
async def test_concurrent_startup_publishes_one_catalog_and_one_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    entered = asyncio.Event()
    release_connect = asyncio.Event()
    owner_ready = asyncio.Event()
    release_owner = asyncio.Event()
    calls = 0

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        entered.set()
        await release_connect.wait()
        return _register_report(loop, stack)

    async def owner() -> None:
        async with loop.mcp_lifespan():
            owner_ready.set()
            await release_owner.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(entered.wait(), timeout=1)
    contender = asyncio.create_task(loop._connect_mcp())
    release_connect.set()
    await asyncio.wait_for(owner_ready.wait(), timeout=1)
    assert await contender is False
    assert calls == 1
    assert len([name for name in loop.tools.tool_names if name.startswith("mcp_")]) == 1

    with pytest.raises(RuntimeError, match="active lifespan"):
        await loop.close_mcp()
    release_owner.set()
    await owner_task
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_run_closes_its_owned_lifespan_when_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    connected = asyncio.Event()
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        report = _register_report(loop, stack)
        connected.set()
        return report

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(connected.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup == ["closed"]
    assert loop._mcp_connected is False
    assert not any(name.startswith("mcp_") for name in loop.tools.tool_names)


@pytest.mark.asyncio
async def test_unrelated_direct_calls_serialize_complete_mcp_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        stack.callback(cleanup.append, f"closed-{calls}")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del session_key
        if msg.content == "first":
            first_entered.set()
            await release_first.wait()
        else:
            second_entered.set()
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=msg.content,
        )

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)

    first = asyncio.create_task(loop.process_direct("first"))
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(loop.process_direct("second"))
    await asyncio.sleep(0)
    assert second_entered.is_set() is False
    assert calls == 1

    release_first.set()
    assert await asyncio.wait_for(first, timeout=1) == "first"
    assert await asyncio.wait_for(second_entered.wait(), timeout=1) is True
    assert await asyncio.wait_for(second, timeout=1) == "second"
    assert calls == 2
    assert cleanup == ["closed-1", "closed-2"]
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_inherited_child_process_reuses_parent_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    calls = 0
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    async with loop.mcp_lifespan():
        child = asyncio.create_task(loop.process_direct("hello"))
        assert await asyncio.wait_for(child, timeout=1) == "fixture"
        assert calls == 1
        assert cleanup == []
        assert loop._mcp_connected is True

    assert cleanup == ["closed"]
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_normal_scope_exit_waits_for_inherited_borrower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    release_borrower = asyncio.Event()
    owner_leaving_body = asyncio.Event()
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del session_key
        borrower_started.set()
        await release_borrower.wait()
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="done",
        )

    async def owner() -> None:
        async with loop.mcp_lifespan():
            asyncio.create_task(loop.process_direct("borrow"))
            await borrower_started.wait()
            owner_leaving_body.set()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_leaving_body.wait(), timeout=1)
    await asyncio.sleep(0)
    assert owner_task.done() is False
    assert cleanup == []
    assert loop._mcp_connected is True

    release_borrower.set()
    await asyncio.wait_for(owner_task, timeout=1)
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}


@pytest.mark.asyncio
async def test_late_inherited_child_fails_instead_of_opening_a_new_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    child_parked = asyncio.Event()
    release_child = asyncio.Event()
    cleanup: list[str] = []
    calls = 0

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        stack.callback(cleanup.append, f"closed-{calls}")
        return _register_report(loop, stack)

    async def late_child() -> str:
        child_parked.set()
        await release_child.wait()
        return await loop.process_direct("late")

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    async with loop.mcp_lifespan():
        child = asyncio.create_task(late_child())
        await child_parked.wait()

    assert cleanup == ["closed-1"]
    release_child.set()
    with pytest.raises(RuntimeError) as exc_info:
        await asyncio.wait_for(child, timeout=1)
    assert str(exc_info.value) == _MCP_INHERITED_SCOPE_REVOKED
    assert calls == 1
    assert cleanup == ["closed-1"]


@pytest.mark.asyncio
async def test_borrower_nested_child_is_rejected_during_drain_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    owner_leaving = asyncio.Event()
    start_nested_child = asyncio.Event()
    nested_errors: list[str] = []
    processed: list[str] = []
    cleanup: list[str] = []
    calls = 0

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del session_key
        processed.append(msg.content)
        borrower_started.set()
        await start_nested_child.wait()
        nested = asyncio.create_task(loop.process_direct("nested"))
        try:
            await nested
        except RuntimeError as exc:
            nested_errors.append(str(exc))
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="borrower finished",
        )

    async def owner() -> None:
        async with loop.mcp_lifespan():
            asyncio.create_task(loop.process_direct("borrower"))
            await borrower_started.wait()
            owner_leaving.set()

    async def wait_for_drain() -> None:
        while loop._mcp_accepting_borrowers:
            await asyncio.sleep(0)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_leaving.wait(), timeout=1)
    await asyncio.wait_for(wait_for_drain(), timeout=1)
    start_nested_child.set()
    await asyncio.wait_for(owner_task, timeout=1)

    assert nested_errors == [_MCP_INHERITED_SCOPE_REVOKED]
    assert processed == ["borrower"]
    assert calls == 1
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}


@pytest.mark.asyncio
async def test_exceptional_scope_exit_cancels_and_reaps_orphan_borrower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    borrower_finished = asyncio.Event()
    never_release = asyncio.Event()
    cleanup: list[str] = []
    orphan: asyncio.Task[str] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        borrower_started.set()
        try:
            await never_release.wait()
        finally:
            borrower_finished.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)

    with pytest.raises(RuntimeError, match="primary failure"):
        async with loop.mcp_lifespan():
            orphan = asyncio.create_task(loop.process_direct("orphan"))
            await borrower_started.wait()
            raise RuntimeError("primary failure")

    assert orphan is not None
    assert orphan.done()
    assert orphan.cancelled()
    assert borrower_finished.is_set()
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_cancelled_owner_reaps_borrower_and_preserves_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    borrower_finished = asyncio.Event()
    owner_blocked = asyncio.Event()
    never_release = asyncio.Event()
    cleanup: list[str] = []
    orphan: asyncio.Task[str] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        borrower_started.set()
        try:
            await never_release.wait()
        finally:
            borrower_finished.set()
        raise AssertionError("unreachable")

    async def owner() -> None:
        nonlocal orphan
        async with loop.mcp_lifespan():
            orphan = asyncio.create_task(loop.process_direct("orphan"))
            await borrower_started.wait()
            owner_blocked.set()
            await never_release.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_blocked.wait(), timeout=1)
    owner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner_task

    assert orphan is not None
    assert orphan.cancelled()
    assert borrower_finished.is_set()
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_double_cancelled_borrower_releases_lease_before_owner_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    borrower_body_finished = asyncio.Event()
    owner_ready = asyncio.Event()
    release_owner = asyncio.Event()
    never_release = asyncio.Event()
    cleanup: list[str] = []
    borrower: asyncio.Task[str] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        borrower_started.set()
        try:
            await never_release.wait()
        finally:
            borrower_body_finished.set()
        raise AssertionError("unreachable")

    async def owner() -> None:
        nonlocal borrower
        async with loop.mcp_lifespan():
            borrower = asyncio.create_task(loop.process_direct("borrower"))
            await borrower_started.wait()
            owner_ready.set()
            await release_owner.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_ready.wait(), timeout=1)
    assert borrower is not None

    await loop._mcp_lifecycle_lock.acquire()
    borrower.cancel()
    await asyncio.wait_for(borrower_body_finished.wait(), timeout=1)
    await asyncio.sleep(0)
    borrower.cancel()
    await asyncio.sleep(0)
    assert borrower.done() is False
    assert borrower in loop._mcp_borrowers
    assert loop._mcp_borrowers_idle.is_set() is False

    release_owner.set()
    await asyncio.sleep(0)
    assert owner_task.done() is False
    loop._mcp_lifecycle_lock.release()

    await asyncio.wait_for(owner_task, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await borrower
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}
    assert loop._mcp_borrowers_idle.is_set()
    assert loop._mcp_active_scope_token is None
    assert loop._mcp_connected is False
    assert loop._mcp_transport_task is None


@pytest.mark.asyncio
async def test_double_cancelled_owner_reaps_borrower_after_drain_lock_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    borrower_finished = asyncio.Event()
    owner_blocked = asyncio.Event()
    never_release = asyncio.Event()
    cleanup: list[str] = []
    borrower: asyncio.Task[str] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        borrower_started.set()
        try:
            await never_release.wait()
        finally:
            borrower_finished.set()
        raise AssertionError("unreachable")

    async def owner() -> None:
        nonlocal borrower
        async with loop.mcp_lifespan():
            borrower = asyncio.create_task(loop.process_direct("borrower"))
            await borrower_started.wait()
            owner_blocked.set()
            await never_release.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_blocked.wait(), timeout=1)
    assert borrower is not None

    await loop._mcp_lifecycle_lock.acquire()
    owner_task.cancel()
    await asyncio.sleep(0)
    owner_task.cancel()
    await asyncio.sleep(0)
    assert owner_task.done() is False
    assert borrower.done() is False
    assert loop._mcp_active_scope_token is not None
    assert loop._mcp_accepting_borrowers is True
    loop._mcp_lifecycle_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner_task, timeout=1)
    assert borrower.cancelled()
    assert borrower_finished.is_set()
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}
    assert loop._mcp_borrowers_idle.is_set()
    assert loop._mcp_active_scope_token is None
    assert loop._mcp_accepting_borrowers is False
    assert loop._mcp_connected is False
    assert loop._mcp_transport_task is None


@pytest.mark.asyncio
async def test_cancel_after_revoke_during_second_lock_wait_aborts_borrower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    borrower_started = asyncio.Event()
    borrower_finished = asyncio.Event()
    owner_ready = asyncio.Event()
    release_owner = asyncio.Event()
    contender_holds_lock = asyncio.Event()
    release_contender = asyncio.Event()
    never_release = asyncio.Event()
    cleanup: list[str] = []
    borrower: asyncio.Task[str] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        borrower_started.set()
        try:
            await never_release.wait()
        finally:
            borrower_finished.set()
        raise AssertionError("unreachable")

    async def owner() -> None:
        nonlocal borrower
        async with loop.mcp_lifespan():
            borrower = asyncio.create_task(loop.process_direct("borrower"))
            await borrower_started.wait()
            owner_ready.set()
            await release_owner.wait()

    async def lock_contender() -> None:
        async with loop._mcp_lifecycle_lock:
            contender_holds_lock.set()
            await release_contender.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_ready.wait(), timeout=1)
    assert borrower is not None

    # Queue the normal owner's revoke first, then a contender.  Lock fairness
    # lets revoke complete before the contender holds the lock, leaving the
    # owner blocked in the drain's second resilient acquisition.
    await loop._mcp_lifecycle_lock.acquire()
    release_owner.set()
    await asyncio.sleep(0)
    contender = asyncio.create_task(lock_contender())
    await asyncio.sleep(0)
    loop._mcp_lifecycle_lock.release()
    await asyncio.wait_for(contender_holds_lock.wait(), timeout=1)
    assert loop._mcp_accepting_borrowers is False
    assert borrower.done() is False

    owner_task.cancel()
    await asyncio.sleep(0)
    assert owner_task.done() is False
    release_contender.set()
    await asyncio.wait_for(contender, timeout=1)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner_task, timeout=1)
    assert borrower.cancelled()
    assert borrower_finished.is_set()
    assert cleanup == ["closed"]
    assert loop._mcp_borrowers == {}
    assert loop._mcp_borrowers_idle.is_set()
    assert loop._mcp_active_scope_token is None
    assert loop._mcp_accepting_borrowers is False
    assert loop._mcp_connected is False
    assert loop._mcp_transport_task is None


@pytest.mark.asyncio
async def test_public_close_is_rejected_without_corrupting_active_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    async with loop.mcp_lifespan():
        with pytest.raises(RuntimeError, match="active lifespan"):
            await asyncio.wait_for(loop.close_mcp(), timeout=1)
        assert loop._mcp_connected is True
        assert any(name.startswith("mcp_") for name in loop.tools.tool_names)
        assert cleanup == []

    assert cleanup == ["closed"]
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_body_and_transport_cleanup_failures_are_both_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failure")

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(fail_cleanup)
        return _register_report(loop, stack)

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    with pytest.raises(BaseExceptionGroup) as captured:
        async with loop.mcp_lifespan():
            raise ValueError("body failure")

    failures = captured.value.exceptions
    assert [type(error) for error in failures] == [ValueError, RuntimeError]
    assert [str(error) for error in failures] == ["body failure", "cleanup failure"]
    assert loop._mcp_connected is False


@pytest.mark.asyncio
async def test_inherited_child_run_reuses_parent_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    calls = 0
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal calls
        del servers, registry
        calls += 1
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def run_connected() -> None:
        child_started.set()
        await release_child.wait()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_run_connected", run_connected)
    async with loop.mcp_lifespan():
        child = asyncio.create_task(loop.run())
        await asyncio.wait_for(child_started.wait(), timeout=1)
        assert calls == 1
        release_child.set()
        await asyncio.wait_for(child, timeout=1)
        assert cleanup == []

    assert cleanup == ["closed"]


@pytest.mark.asyncio
async def test_cancelled_child_direct_call_uses_one_transport_owner_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    processing = asyncio.Event()
    never_release = asyncio.Event()
    connect_task: asyncio.Task[Any] | None = None
    cleanup_task: asyncio.Task[Any] | None = None

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal connect_task
        del servers, registry
        connect_task = asyncio.current_task()

        def record_cleanup_task() -> None:
            nonlocal cleanup_task
            cleanup_task = asyncio.current_task()

        stack.callback(record_cleanup_task)
        return _register_report(loop, stack)

    async def process_message(
        msg: Any,
        session_key: str | None = None,
    ) -> OutboundMessage:
        del msg, session_key
        processing.set()
        await never_release.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    monkeypatch.setattr(loop, "_process_message", process_message)
    child = asyncio.create_task(loop.process_direct("cancel me"))
    await asyncio.wait_for(processing.wait(), timeout=1)
    child.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child

    assert connect_task is not None
    assert connect_task is cleanup_task
    assert connect_task is not child
    assert loop._mcp_connected is False
    assert not any(name.startswith("mcp_") for name in loop.tools.tool_names)


@pytest.mark.asyncio
async def test_cancellation_during_close_waits_for_same_task_transport_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    connect_tasks: list[asyncio.Task[Any] | None] = []
    close_tasks: list[asyncio.Task[Any] | None] = []
    attempts = 0

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        nonlocal attempts
        del servers, registry
        attempts += 1
        connect_tasks.append(asyncio.current_task())

        async def slow_close() -> None:
            close_tasks.append(asyncio.current_task())
            close_started.set()
            try:
                await release_close.wait()
            finally:
                close_finished.set()

        stack.push_async_callback(slow_close)
        return _register_report(loop, stack)

    async def own_one_scope() -> None:
        async with loop.mcp_lifespan():
            pass

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    owner = asyncio.create_task(own_one_scope())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    transport_owner = loop._mcp_transport_task
    assert transport_owner is not None
    owner.cancel()
    await asyncio.sleep(0)
    owner.cancel()
    await asyncio.sleep(0)
    assert owner.done() is False
    assert transport_owner.cancelling() == 0
    assert loop._mcp_connected is True
    assert loop._mcp_stack is not None

    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=1)

    assert close_finished.is_set()
    assert connect_tasks == close_tasks
    assert connect_tasks[0] is not owner
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert loop._mcp_transport_task is None
    assert not any(name.startswith("mcp_") for name in loop.tools.tool_names)

    # Once the first owner has truthfully finished, a fresh scope is usable.
    close_started.clear()
    close_finished.clear()
    await asyncio.wait_for(own_one_scope(), timeout=1)
    assert attempts == 2
    assert connect_tasks == close_tasks
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_close_cancellation_while_state_lock_is_held_still_reaps_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path)
    connected = asyncio.Event()
    allow_close = asyncio.Event()
    close_attempted = asyncio.Event()
    cleanup: list[str] = []

    async def connect(
        servers: dict[str, Any],
        registry: Any,
        stack: AsyncExitStack,
    ) -> MCPConnectionReport:
        del servers, registry
        stack.callback(cleanup.append, "closed")
        return _register_report(loop, stack)

    async def own_manual_connection() -> None:
        assert await loop._connect_mcp() is True
        connected.set()
        await allow_close.wait()
        close_attempted.set()
        await loop.close_mcp()

    monkeypatch.setattr("k2do.agent.tools.mcp.connect_mcp_servers", connect)
    await loop._mcp_lifecycle_lock.acquire()
    owner = asyncio.create_task(own_manual_connection())
    loop._mcp_lifecycle_lock.release()
    await asyncio.wait_for(connected.wait(), timeout=1)

    await loop._mcp_lifecycle_lock.acquire()
    allow_close.set()
    await asyncio.wait_for(close_attempted.wait(), timeout=1)
    await asyncio.sleep(0)
    owner.cancel()
    await asyncio.sleep(0)
    assert owner.done() is False
    assert loop._mcp_connected is True

    loop._mcp_lifecycle_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=1)

    assert cleanup == ["closed"]
    assert loop._mcp_connected is False
    assert loop._mcp_stack is None
    assert loop._mcp_transport_task is None


@pytest.mark.asyncio
async def test_background_services_aclose_awaits_cancelled_workers(
    tmp_path: Path,
) -> None:
    heartbeat = HeartbeatService(tmp_path, interval_s=3600)
    await heartbeat.start()
    heartbeat_task = heartbeat._task
    assert heartbeat_task is not None
    await heartbeat.start()
    assert heartbeat._task is heartbeat_task

    cron = CronService(tmp_path / "cron.json")
    cron.add_job(
        name="fixture",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="fixture",
    )
    await cron.start()
    cron_task = cron._timer_task
    assert cron_task is not None
    await cron.start()
    assert cron._timer_task is cron_task

    await heartbeat.aclose()
    await cron.aclose()
    assert heartbeat_task.done()
    assert cron_task.done()
    assert heartbeat._task is None
    assert cron._timer_task is None
    assert heartbeat._running is False
    assert cron._running is False

    # Both shutdown APIs are intentionally idempotent.
    await heartbeat.aclose()
    await cron.aclose()


@pytest.mark.asyncio
async def test_service_restart_reaps_previous_worker_before_replacement(
    tmp_path: Path,
) -> None:
    heartbeat = HeartbeatService(tmp_path, interval_s=3600)
    await heartbeat.start()
    old_heartbeat = heartbeat._task
    assert old_heartbeat is not None
    heartbeat.stop()
    await heartbeat.start()
    assert old_heartbeat.done()
    assert heartbeat._task is not old_heartbeat

    cron = CronService(tmp_path / "cron-restart.json")
    cron.add_job(
        name="fixture",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="fixture",
    )
    await cron.start()
    old_cron = cron._timer_task
    assert old_cron is not None
    cron.stop()
    await cron.start()
    assert old_cron.done()
    assert cron._timer_task is not old_cron

    await heartbeat.aclose()
    await cron.aclose()


@pytest.mark.asyncio
async def test_heartbeat_aclose_does_not_hide_worker_failures(
    tmp_path: Path,
) -> None:
    heartbeat = HeartbeatService(tmp_path)

    async def fail() -> None:
        raise RuntimeError("worker failed")

    task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    heartbeat._task = task
    with pytest.raises(RuntimeError, match="worker failed"):
        await heartbeat.aclose()
    assert heartbeat._task is None


@pytest.mark.asyncio
async def test_service_aclose_preserves_caller_cancellation(
    tmp_path: Path,
) -> None:
    services_and_attrs = (
        (HeartbeatService(tmp_path), "_task"),
        (CronService(tmp_path / "cron-cancel.json"), "_timer_task"),
    )

    for service, task_attr in services_and_attrs:
        worker_saw_first_cancel = asyncio.Event()
        release_worker_cleanup = asyncio.Event()

        async def stubborn_worker(
            started: asyncio.Event,
            release: asyncio.Event,
        ) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                started.set()
                await release.wait()

        worker = asyncio.create_task(
            stubborn_worker(worker_saw_first_cancel, release_worker_cleanup)
        )
        setattr(service, task_attr, worker)
        closer = asyncio.create_task(service.aclose())
        await asyncio.wait_for(worker_saw_first_cancel.wait(), timeout=1)
        closer.cancel()
        closer.cancel()
        await asyncio.sleep(0)
        assert closer.done() is False
        assert worker.done() is False
        assert getattr(service, task_attr) is worker

        release_worker_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closer, timeout=1)
        assert worker.done()
        assert getattr(service, task_attr) is None


@pytest.mark.asyncio
async def test_agent_aclose_reaps_owned_background_tasks(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    consolidation_started = asyncio.Event()
    subagent_started = asyncio.Event()
    consolidation_finished = asyncio.Event()
    subagent_finished = asyncio.Event()
    never_release = asyncio.Event()

    async def background(started: asyncio.Event, finished: asyncio.Event) -> None:
        started.set()
        try:
            await never_release.wait()
        finally:
            finished.set()

    consolidation = asyncio.create_task(
        background(consolidation_started, consolidation_finished)
    )
    subagent = asyncio.create_task(background(subagent_started, subagent_finished))
    loop._consolidation_tasks["fixture"] = consolidation
    loop.subagents._running_tasks["fixture"] = subagent
    await asyncio.wait_for(consolidation_started.wait(), timeout=1)
    await asyncio.wait_for(subagent_started.wait(), timeout=1)

    await loop.aclose()
    assert consolidation.done()
    assert subagent.done()
    assert consolidation_finished.is_set()
    assert subagent_finished.is_set()
    assert loop._consolidation_tasks == {}
    assert loop.subagents._running_tasks == {}


@pytest.mark.asyncio
async def test_gateway_cleanup_runs_every_step_and_groups_all_failures() -> None:
    calls: list[str] = []
    primary = ValueError("body failed")
    first_cleanup_error = RuntimeError("heartbeat failed")
    second_cleanup_error = KeyboardInterrupt("channel cleanup interrupted")

    def fail_sync() -> None:
        calls.append("sync")
        raise first_cleanup_error

    async def succeed_async() -> None:
        calls.append("async-success")

    async def fail_async() -> None:
        calls.append("async-failure")
        raise second_cleanup_error

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await _run_gateway_cleanup(
            primary,
            (
                ("sync", fail_sync),
                ("async-success", succeed_async),
                ("async-failure", fail_async),
            ),
        )

    assert calls == ["sync", "async-success", "async-failure"]
    assert exc_info.value.exceptions == (
        primary,
        first_cleanup_error,
        second_cleanup_error,
    )


@pytest.mark.asyncio
async def test_gateway_cleanup_preserves_plain_caller_cancellation() -> None:
    first_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    calls: list[str] = []

    async def blocking_cleanup() -> None:
        calls.append("blocking-start")
        first_started.set()
        try:
            await release_cleanup.wait()
        finally:
            calls.append("blocking-finished")

    async def remaining_cleanup() -> None:
        calls.append("remaining")

    cleanup_task = asyncio.create_task(
        _run_gateway_cleanup(
            None,
            (
                ("blocking", blocking_cleanup),
                ("remaining", remaining_cleanup),
            ),
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    cleanup_task.cancel()
    cleanup_task.cancel()
    await asyncio.sleep(0)
    assert cleanup_task.done() is False
    assert calls == ["blocking-start"]

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cleanup_task, timeout=1)

    assert calls == ["blocking-start", "blocking-finished", "remaining"]


def test_mcp_tool_logging_never_emits_argument_values() -> None:
    messages: list[str] = []
    sink = __import__("loguru").logger.add(messages.append, format="{message}")
    try:
        _log_tool_call(
            "mcp_0123456789abcdef0123456789abcdef",
            {"credential": "PRIVATE-CANARY-VALUE"},
            inline=False,
        )
    finally:
        __import__("loguru").logger.remove(sink)
    rendered = "".join(messages)
    assert "PRIVATE-CANARY-VALUE" not in rendered
    assert "values redacted" in rendered


def test_malformed_mcp_tool_logging_uses_a_fixed_marker() -> None:
    malformed = "mcp_\nIGNORE-LOG-CANARY"
    messages: list[str] = []
    sink = __import__("loguru").logger.add(messages.append, format="{message}")
    try:
        _log_tool_call(malformed, {"credential": "PRIVATE-CANARY-VALUE"}, inline=True)
    finally:
        __import__("loguru").logger.remove(sink)

    rendered = "".join(messages)
    assert MCP_INVALID_NAME_MARKER in rendered
    assert malformed not in rendered
    assert "LOG-CANARY" not in rendered
    assert "PRIVATE-CANARY-VALUE" not in rendered


@pytest.mark.asyncio
async def test_provider_failure_log_never_includes_exception_text(tmp_path: Path) -> None:
    canary = "PRIVATE-PROVIDER-EXCEPTION-CANARY"

    class FailingProvider(_Provider):
        async def chat(self, **kwargs: Any) -> LLMResponse:
            del kwargs
            raise RuntimeError(canary)

    loop = _loop_without_mcp(tmp_path)
    loop.provider = FailingProvider(api_key=None, api_base=None)
    messages: list[str] = []
    sink = __import__("loguru").logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(RuntimeError, match="backend unavailable"):
            await loop._chat_with_fallback([], "offline/fixture", tools=[])
    finally:
        __import__("loguru").logger.remove(sink)

    rendered = "".join(messages)
    assert canary not in rendered
    assert "RuntimeError" in rendered
