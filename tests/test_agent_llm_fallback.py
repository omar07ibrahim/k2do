from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _FailThenSucceedProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
        self.calls: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        selected = model or self.get_default_model()
        self.calls.append(selected)
        if "instruct" in selected.lower():
            raise RuntimeError("primary path failed")
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


class _AlwaysFailProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        raise RuntimeError("backend down")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_agent_loop_falls_back_to_second_model(tmp_path: Path) -> None:
    provider = _FailThenSucceedProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    response = await loop._chat_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        model="k2-v2-instruct/LLM360/K2-V2-Instruct",
    )

    assert response.content == "ok"
    assert provider.calls[:3] == [
        "k2-v2-instruct/LLM360/K2-V2-Instruct",
        "k2-v2-instruct/LLM360/K2-V2-Instruct",
        "k2-think-v2/LLM360/K2-Think-V2",
    ]


@pytest.mark.asyncio
async def test_process_direct_returns_generic_error_on_total_failure(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_AlwaysFailProvider(),
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    text = await loop.process_direct("hello")
    assert "internal error" in text.lower()
