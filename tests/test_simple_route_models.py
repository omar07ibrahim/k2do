from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _CaptureModelProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
        self.models: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = messages, tools, max_tokens, temperature
        self.models.append(model or self.get_default_model())
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_simple_greeting_prefers_fast_model(tmp_path: Path) -> None:
    provider = _CaptureModelProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    response = await loop.process_direct("hello")

    assert response == "ok"
    assert provider.models[0] == "k2-v2-instruct/LLM360/K2-V2-Instruct"


@pytest.mark.asyncio
async def test_action_request_prefers_think_model(tmp_path: Path) -> None:
    provider = _CaptureModelProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    response = await loop.process_direct("создай файл x.txt")

    assert response == "ok"
    assert provider.models[0] == "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_last_route_is_simple_when_deepthink_disabled(tmp_path: Path) -> None:
    provider = _CaptureModelProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    response = await loop.process_direct("Сгенерируй трек в арабском стиле")

    assert response == "ok"
    assert loop._last_route == "simple"
