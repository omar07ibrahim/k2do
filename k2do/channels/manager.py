"""Channel manager for K2DO."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from k2do.bus.queue import MessageBus
from k2do.channels.base import BaseChannel
from k2do.config.schema import Config


class ChannelManager:
    """Manages chat channels and coordinates message routing."""

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._dispatch_cancel_requested_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task[BaseException | None] | None = None
        self._generation = 0
        self._stopped_generation = -1
        self._init_channels()

    def _init_channels(self) -> None:
        if self.config.channels.telegram.enabled:
            try:
                from k2do.channels.telegram import TelegramChannel
                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning(f"Telegram channel not available: {e}")

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        try:
            await channel.start()
        except Exception as e:
            logger.error(f"Failed to start channel {name}: {e}")

    async def start_all(self) -> None:
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # A shutdown owner has precedence over a new generation.  In
        # particular, a failed owner leaves ``_stopped_generation`` behind;
        # the next start must retry that cleanup instead of replacing live
        # handles from the incomplete generation.
        stop_task = self._stop_task
        if stop_task is not None:
            await self._await_stop_task(stop_task)

        dispatch_task = self._dispatch_task
        if dispatch_task is not None and not dispatch_task.done():
            logger.debug("Channels already running")
            return

        if self._generation > 0 and self._stopped_generation < self._generation:
            await self.stop_all()

        # Concurrent starters can both have awaited the same stop owner.  The
        # first one publishes its dispatcher before its first subsequent
        # await; the second must observe and reuse that active generation.
        dispatch_task = self._dispatch_task
        if dispatch_task is not None and not dispatch_task.done():
            logger.debug("Channels already running")
            return

        self._generation += 1
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())
        self._dispatch_cancel_requested_task = None
        tasks = []
        for name, channel in self.channels.items():
            logger.info(f"Starting {name} channel...")
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        logger.info("Stopping all channels...")
        stop_task = self._stop_task
        if stop_task is None:
            if self._stopped_generation == self._generation:
                return
            generation = self._generation
            stop_task = asyncio.create_task(
                self._stop_owner(
                    generation,
                    self._dispatch_task,
                    tuple(self.channels.items()),
                )
            )
            self._stop_task = stop_task
        await self._await_stop_task(stop_task)

    async def _stop_owner(
        self,
        generation: int,
        dispatch_task: asyncio.Task | None,
        channels: tuple[tuple[str, BaseChannel], ...],
    ) -> BaseException | None:
        errors: list[BaseException] = []
        if dispatch_task is not None:
            if (
                not dispatch_task.done()
                and dispatch_task.cancelling() == 0
                and self._dispatch_cancel_requested_task is not dispatch_task
            ):
                self._dispatch_cancel_requested_task = dispatch_task
                dispatch_task.cancel()
            try:
                await dispatch_task
            except asyncio.CancelledError as exc:
                if self._dispatch_cancel_requested_task is not dispatch_task:
                    exc.add_note("Outbound dispatch stopped independently")
                    errors.append(exc)
            except BaseException as exc:
                exc.add_note("Outbound dispatch failed during channel shutdown")
                errors.append(exc)
            finally:
                if self._dispatch_task is dispatch_task:
                    self._dispatch_task = None
                if self._dispatch_cancel_requested_task is dispatch_task:
                    self._dispatch_cancel_requested_task = None

        for name, channel in channels:
            try:
                await channel.stop()
            except BaseException as exc:
                exc.add_note(f"Channel shutdown failed: {name}")
                errors.append(exc)
                logger.error(f"Error stopping {name}: {type(exc).__name__}")

        if len(errors) == 1:
            return errors[0]
        if errors:
            return BaseExceptionGroup("Channel shutdown failed", errors)
        self._stopped_generation = max(self._stopped_generation, generation)
        return None

    async def _await_stop_task(
        self,
        stop_task: asyncio.Task[BaseException | None],
    ) -> None:
        caller_cancellation: asyncio.CancelledError | None = None
        current = asyncio.current_task()
        while not stop_task.done():
            cancelling_before = current.cancelling() if current is not None else 0
            try:
                await asyncio.shield(stop_task)
            except asyncio.CancelledError as exc:
                cancelling_after = current.cancelling() if current is not None else 0
                if not stop_task.done() or cancelling_after > cancelling_before:
                    if caller_cancellation is None:
                        caller_cancellation = exc
                    continue
                break
            except BaseException:
                break

        try:
            stop_error = stop_task.result()
        except BaseException as exc:
            stop_error = exc
        if self._stop_task is stop_task:
            self._stop_task = None

        if caller_cancellation is not None and stop_error is not None:
            raise BaseExceptionGroup(
                "Channel shutdown failed after caller cancellation",
                (caller_cancellation, stop_error),
            ) from None
        if stop_error is not None:
            raise stop_error
        if caller_cancellation is not None:
            raise caller_cancellation

    async def _dispatch_outbound(self) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)
                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error(f"Error sending to {msg.channel}: {e}")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> BaseChannel | None:
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())
