import asyncio
from pathlib import Path

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse
from k2do.session.manager import Session


class _NoopProvider(LLMProvider):
    async def chat(self, *args, **kwargs) -> LLMResponse:  # type: ignore[override]
        _ = args, kwargs
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_schedule_consolidation_deduplicates_running_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_NoopProvider(),
        workspace=tmp_path,
        deepthink_enabled=False,
    )

    session = Session(key="cli:test")
    calls = {"count": 0}

    async def fake_consolidate(_session, archive_all: bool = False) -> None:
        _ = _session, archive_all
        calls["count"] += 1
        await asyncio.sleep(0.05)

    monkeypatch.setattr(loop, "_consolidate_memory", fake_consolidate)

    loop._schedule_consolidation(session)
    loop._schedule_consolidation(session)
    await asyncio.sleep(0.12)

    assert calls["count"] == 1
