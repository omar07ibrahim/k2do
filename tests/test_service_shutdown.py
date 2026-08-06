from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from k2do.bus.queue import MessageBus
from k2do.channels.manager import ChannelManager
from k2do.channels.telegram import TelegramChannel
from k2do.cli.commands import _run_gateway_cleanup
from k2do.cron.service import CronService
from k2do.cron.types import CronSchedule
from k2do.heartbeat.service import HeartbeatService


def _service(tmp_path: Path, kind: str) -> tuple[Any, str]:
    if kind == "heartbeat":
        return HeartbeatService(tmp_path, interval_s=3600), "_task"
    cron = CronService(tmp_path / "cron.json")
    cron.add_job(
        name="fixture",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="fixture",
    )
    return cron, "_timer_task"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["heartbeat", "cron"])
async def test_concurrent_service_close_uses_one_shielded_drain(
    tmp_path: Path,
    kind: str,
) -> None:
    service, task_attr = _service(tmp_path, kind)
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    resource_closed = asyncio.Event()
    never_release = asyncio.Event()

    async def worker_body() -> None:
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            finalizer_started.set()
            await release_finalizer.wait()
            resource_closed.set()

    worker = asyncio.create_task(worker_body())
    setattr(service, task_attr, worker)
    service._running = True
    first = asyncio.create_task(service.aclose())
    await asyncio.wait_for(finalizer_started.wait(), timeout=1)
    second = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)

    first.cancel()
    first.cancel()
    await asyncio.sleep(0)
    assert worker.cancelling() == 1
    assert worker.done() is False
    assert getattr(service, task_attr) is worker
    assert resource_closed.is_set() is False

    release_finalizer.set()
    await asyncio.wait_for(second, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=1)
    assert resource_closed.is_set()
    assert worker.done()
    assert getattr(service, task_attr) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["heartbeat", "cron"])
async def test_cancelled_service_restart_drains_old_worker_without_replacement(
    tmp_path: Path,
    kind: str,
) -> None:
    service, task_attr = _service(tmp_path, kind)
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    resource_closed = asyncio.Event()
    never_release = asyncio.Event()

    async def worker_body() -> None:
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            finalizer_started.set()
            await release_finalizer.wait()
            resource_closed.set()

    worker = asyncio.create_task(worker_body())
    setattr(service, task_attr, worker)
    service._running = True
    await asyncio.sleep(0)
    service.stop()
    await asyncio.wait_for(finalizer_started.wait(), timeout=1)

    restart = asyncio.create_task(service.start())
    await asyncio.sleep(0)
    restart.cancel()
    restart.cancel()
    await asyncio.sleep(0)
    assert worker.cancelling() == 1
    assert getattr(service, task_attr) is worker

    release_finalizer.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(restart, timeout=1)
    assert resource_closed.is_set()
    assert getattr(service, task_attr) is None
    assert service._running is False

    await service.start()
    assert service._running is True
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["heartbeat", "cron"])
async def test_service_does_not_recancel_externally_cancelling_worker(
    tmp_path: Path,
    kind: str,
) -> None:
    service, task_attr = _service(tmp_path, kind)
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    resource_closed = asyncio.Event()

    async def worker_body() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finalizer_started.set()
            await release_finalizer.wait()
            resource_closed.set()

    worker = asyncio.create_task(worker_body())
    setattr(service, task_attr, worker)
    service._running = True
    await asyncio.sleep(0)
    worker.cancel()
    await asyncio.wait_for(finalizer_started.wait(), timeout=1)

    closer = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert worker.cancelling() == 1
    assert worker.done() is False
    assert getattr(service, task_attr) is worker

    release_finalizer.set()
    await asyncio.wait_for(closer, timeout=1)
    assert resource_closed.is_set()
    assert getattr(service, task_attr) is None


@pytest.mark.asyncio
async def test_cron_start_reaps_failed_timer_before_restart(tmp_path: Path) -> None:
    cron = CronService(tmp_path / "cron-failed.json")

    async def fail() -> None:
        raise RuntimeError("timer failed")

    failed = asyncio.create_task(fail())
    await asyncio.sleep(0)
    cron._running = True
    cron._timer_task = failed
    with pytest.raises(RuntimeError, match="timer failed"):
        await cron.start()
    assert cron._running is False
    assert cron._timer_task is None

    cron.add_job(
        name="fixture",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="fixture",
    )
    await cron.start()
    assert cron._timer_task is not None
    await cron.aclose()


@pytest.mark.asyncio
async def test_gateway_cleanup_survives_repeated_caller_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    calls: list[str] = []

    async def held_cleanup() -> None:
        calls.append("held-start")
        started.set()
        await release.wait()
        closed.set()
        calls.append("held-finished")

    async def later_cleanup() -> None:
        calls.append("later")

    runner = asyncio.create_task(
        _run_gateway_cleanup(
            None,
            (("held", held_cleanup), ("later", later_cleanup)),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    runner.cancel()
    runner.cancel()
    await asyncio.sleep(0)
    assert runner.done() is False
    assert closed.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(runner, timeout=1)
    assert closed.is_set()
    assert calls == ["held-start", "held-finished", "later"]


@pytest.mark.asyncio
async def test_gateway_distinguishes_child_and_caller_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleanup_error = RuntimeError("cleanup failed")

    async def held_failure() -> None:
        started.set()
        await release.wait()
        raise cleanup_error

    async def self_cancel() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)

    runner = asyncio.create_task(
        _run_gateway_cleanup(
            ValueError("body failed"),
            (("held", held_failure), ("self-cancel", self_cancel)),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    runner.cancel()
    runner.cancel()
    release.set()
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(runner, timeout=1)

    failures = exc_info.value.exceptions
    assert [type(item) for item in failures] == [
        ValueError,
        asyncio.CancelledError,
        RuntimeError,
        asyncio.CancelledError,
    ]
    assert failures[2] is cleanup_error


def _manager(
    dispatch: asyncio.Task[None],
    channels: dict[str, Any],
) -> ChannelManager:
    manager = object.__new__(ChannelManager)
    manager.channels = channels
    manager._dispatch_task = dispatch
    manager._dispatch_cancel_requested_task = None
    manager._stop_task = None
    manager._generation = 1
    manager._stopped_generation = 0
    return manager


def _unstarted_manager(channels: dict[str, Any]) -> ChannelManager:
    manager = object.__new__(ChannelManager)
    manager.bus = MessageBus()
    manager.channels = channels
    manager._dispatch_task = None
    manager._dispatch_cancel_requested_task = None
    manager._stop_task = None
    manager._generation = 0
    manager._stopped_generation = -1
    return manager


@pytest.mark.asyncio
async def test_channel_manager_double_and_concurrent_start_reuses_generation() -> None:
    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    calls: list[str] = []

    class Channel:
        async def start(self) -> None:
            calls.append("start")
            start_entered.set()
            await release_start.wait()

        async def stop(self) -> None:
            calls.append("stop")

    manager = _unstarted_manager({"fixture": Channel()})
    first = asyncio.create_task(manager.start_all())
    await asyncio.wait_for(start_entered.wait(), timeout=1)
    dispatch = manager._dispatch_task
    assert dispatch is not None

    await asyncio.wait_for(manager.start_all(), timeout=1)
    assert manager._dispatch_task is dispatch
    assert calls == ["start"]

    release_start.set()
    await asyncio.wait_for(first, timeout=1)
    await manager.start_all()
    assert manager._dispatch_task is dispatch
    assert calls == ["start"]

    await manager.stop_all()
    assert dispatch.done()
    assert calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_channel_manager_start_retries_failed_generation_cleanup() -> None:
    calls: list[str] = []
    fail_first_stop = True

    class Channel:
        async def start(self) -> None:
            calls.append("start")

        async def stop(self) -> None:
            nonlocal fail_first_stop
            calls.append("stop")
            if fail_first_stop:
                fail_first_stop = False
                raise RuntimeError("first stop failed")

    manager = _unstarted_manager({"fixture": Channel()})
    await manager.start_all()
    failed_generation = manager._generation

    with pytest.raises(RuntimeError, match="first stop failed"):
        await manager.stop_all()
    assert manager._stopped_generation < failed_generation

    await manager.start_all()
    assert calls == ["start", "stop", "stop", "start"]
    assert manager._stopped_generation == failed_generation
    assert manager._generation == failed_generation + 1
    await manager.stop_all()


@pytest.mark.asyncio
async def test_channel_manager_stop_is_singleton_and_cancellation_safe() -> None:
    release_dispatch = asyncio.Event()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls: list[str] = []

    async def dispatch_body() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_dispatch.wait()

    class Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            calls.append(self.name)
            if self.name == "first":
                first_started.set()
                await release_first.wait()

    dispatch = asyncio.create_task(dispatch_body())
    manager = _manager(
        dispatch,
        {"first": Channel("first"), "second": Channel("second")},
    )
    first = asyncio.create_task(manager.stop_all())
    second = asyncio.create_task(manager.stop_all())
    release_dispatch.set()
    await asyncio.wait_for(first_started.wait(), timeout=1)
    first.cancel()
    first.cancel()
    await asyncio.sleep(0)
    assert calls == ["first"]
    assert manager._dispatch_task is None

    release_first.set()
    await asyncio.wait_for(second, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=1)
    assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_channel_manager_attempts_all_stops_before_raising() -> None:
    calls: list[str] = []
    failure = RuntimeError("first failed")
    should_fail = True

    class Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            nonlocal should_fail
            calls.append(self.name)
            if self.name == "first" and should_fail:
                should_fail = False
                raise failure

    async def finished_dispatch() -> None:
        return None

    manager = _manager(
        asyncio.create_task(finished_dispatch()),
        {"first": Channel("first"), "second": Channel("second")},
    )
    with pytest.raises(RuntimeError, match="first failed"):
        await manager.stop_all()
    assert calls == ["first", "second"]
    assert manager._dispatch_task is None
    await manager.stop_all()
    assert calls == ["first", "second", "first", "second"]


@pytest.mark.asyncio
async def test_telegram_stop_attempts_every_terminal_step() -> None:
    updater_started = asyncio.Event()
    release_updater = asyncio.Event()
    calls: list[str] = []
    failure = RuntimeError("application stop failed")
    should_fail = True

    class Updater:
        async def stop(self) -> None:
            calls.append("updater")
            updater_started.set()
            await release_updater.wait()

    async def app_stop() -> None:
        nonlocal should_fail
        calls.append("app")
        if should_fail:
            should_fail = False
            raise failure

    async def app_shutdown() -> None:
        calls.append("shutdown")

    app = SimpleNamespace(
        updater=Updater(),
        stop=app_stop,
        shutdown=app_shutdown,
    )
    channel = object.__new__(TelegramChannel)
    channel._running = True
    channel._app = app
    channel._typing_tasks = {}
    channel._stop_task = None

    first = asyncio.create_task(channel.stop())
    second = asyncio.create_task(channel.stop())
    await asyncio.wait_for(updater_started.wait(), timeout=1)
    first.cancel()
    first.cancel()
    release_updater.set()

    with pytest.raises(RuntimeError, match="application stop failed"):
        await asyncio.wait_for(second, timeout=1)
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(first, timeout=1)
    assert [type(item) for item in exc_info.value.exceptions] == [
        asyncio.CancelledError,
        RuntimeError,
    ]
    assert calls == ["updater", "app", "shutdown"]
    assert channel._app is app

    await channel.stop()
    assert calls == ["updater", "app", "shutdown", "updater", "app", "shutdown"]
    assert channel._app is None


@pytest.mark.asyncio
async def test_telegram_start_retries_retained_app_cleanup_before_replacement() -> None:
    calls: list[str] = []
    failure = RuntimeError("retained application stop failed")

    class Updater:
        async def stop(self) -> None:
            calls.append("updater")

    async def app_stop() -> None:
        calls.append("app")
        raise failure

    async def app_shutdown() -> None:
        calls.append("shutdown")

    old_app = SimpleNamespace(
        updater=Updater(),
        stop=app_stop,
        shutdown=app_shutdown,
    )
    channel = object.__new__(TelegramChannel)
    channel.config = SimpleNamespace(token="fixture-token", proxy=None)
    channel._running = False
    channel._app = old_app
    channel._typing_tasks = {}
    channel._start_task = None
    channel._stop_task = None

    with pytest.raises(RuntimeError, match="retained application stop failed"):
        await channel.stop()
    assert channel._app is old_app

    with pytest.raises(RuntimeError, match="retained application stop failed"):
        await channel.start()
    assert calls == [
        "updater",
        "app",
        "shutdown",
        "updater",
        "app",
        "shutdown",
    ]
    assert channel._app is old_app
    assert channel._start_task is None


@pytest.mark.asyncio
async def test_telegram_restart_waits_for_previous_start_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    builder_called = asyncio.Event()

    async def old_start_owner() -> None:
        old_started.set()
        await release_old.wait()

    def fail_after_ownership_transfer() -> None:
        builder_called.set()
        raise RuntimeError("replacement builder reached")

    old_owner = asyncio.create_task(old_start_owner())
    await asyncio.wait_for(old_started.wait(), timeout=1)

    channel = object.__new__(TelegramChannel)
    channel.config = SimpleNamespace(token="fixture-token", proxy=None)
    channel._running = False
    channel._app = None
    channel._typing_tasks = {}
    channel._start_task = old_owner
    channel._stop_task = None
    monkeypatch.setattr(
        "k2do.channels.telegram.Application.builder",
        fail_after_ownership_transfer,
    )

    replacement = asyncio.create_task(channel.start())
    await asyncio.sleep(0)
    assert replacement.done() is False
    assert builder_called.is_set() is False

    release_old.set()
    with pytest.raises(RuntimeError, match="replacement builder reached"):
        await asyncio.wait_for(replacement, timeout=1)
    assert builder_called.is_set()
    assert channel._start_task is None
