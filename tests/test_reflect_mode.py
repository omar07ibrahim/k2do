from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _NoopProvider(LLMProvider):
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
        _ = messages, tools, model, max_tokens, temperature
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


class _ToolThenDoneProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
        self.followup_user_prompts: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = tools, model, max_tokens, temperature
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if not has_tool_result:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="tc1", name="list_dir", arguments={"path": "."})],
            )

        user_prompts = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
        self.followup_user_prompts.append(user_prompts[-1] if user_prompts else "")
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_reflect_command_switches_mode(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_NoopProvider(),
        workspace=tmp_path,
        deepthink_enabled=False,
    )

    assert await loop.process_direct("/reflect status") == "Reflect mode: OFF"
    assert await loop.process_direct("/reflect on") == "Reflect mode: ON"
    assert loop.sessions.get_or_create("cli:direct").metadata.get("reflect_mode") is True
    assert await loop.process_direct("/reflect off") == "Reflect mode: OFF"
    assert loop.sessions.get_or_create("cli:direct").metadata.get("reflect_mode") is False


@pytest.mark.asyncio
async def test_reflect_mode_changes_followup_prompt(tmp_path: Path) -> None:
    provider = _ToolThenDoneProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        deepthink_enabled=False,
    )

    response_1 = await loop.process_direct("покажи файлы")
    assert response_1 == "done"
    assert "Do not output planning/reflection sections." in provider.followup_user_prompts[-1]

    assert await loop.process_direct("/reflect on") == "Reflect mode: ON"

    response_2 = await loop.process_direct("покажи файлы снова")
    assert response_2 == "done"
    assert provider.followup_user_prompts[-1] == "Reflect on the results and decide next steps."
