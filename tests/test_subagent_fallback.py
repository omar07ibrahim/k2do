from pathlib import Path
from typing import Any

import pytest

from k2do.agent.subagent import SubagentManager
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _SubagentFailThenSucceedProvider(LLMProvider):
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
        _ = messages, tools, max_tokens, temperature
        selected = model or self.get_default_model()
        self.calls.append(selected)
        if "think" in selected:
            raise RuntimeError("temporary backend error")
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_subagent_chat_uses_fallback_model(tmp_path: Path) -> None:
    provider = _SubagentFailThenSucceedProvider()
    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
    )

    response = await manager._chat_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="k2-think-v2/LLM360/K2-Think-V2",
    )

    assert response.content == "ok"
    assert provider.calls[-1] == "k2-v2-instruct/LLM360/K2-V2-Instruct"
