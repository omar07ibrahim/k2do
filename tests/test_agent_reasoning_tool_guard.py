from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _RepeatDeepthinkProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
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
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="d1", name="deepthink", arguments={"question": "plan"})],
            )
        if self.calls == 2:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="d2", name="deepthink", arguments={"question": "plan again"})],
            )
        if self.calls == 3:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="w1", name="write_file", arguments={"path": "x.txt", "content": "ok"})],
            )
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


class _DeepthinkWithConcreteBetweenProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
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
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="d1", name="deepthink", arguments={"question": "plan 1"})],
            )
        if self.calls == 2:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="w1", name="write_file", arguments={"path": "a.txt", "content": "ok"})],
            )
        if self.calls == 3:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="d2", name="deepthink", arguments={"question": "plan 2"})],
            )
        if self.calls == 4:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="e1", name="exec", arguments={"command": "echo ok"})],
            )
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_only_one_reasoning_tool_call_allowed_per_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _RepeatDeepthinkProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    executed: list[str] = []

    async def fake_execute(name: str, params: dict[str, Any]) -> str:
        _ = params
        executed.append(name)
        return "ok"

    monkeypatch.setattr(loop.tools, "execute", fake_execute)

    final, tools_used = await loop._run_agent_loop(
        initial_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        model="k2-think-v2/LLM360/K2-Think-V2",
        allow_reasoning_tools=True,
    )

    assert final == "done"
    assert executed.count("deepthink") == 1
    assert executed.count("write_file") == 1
    assert tools_used == ["deepthink", "write_file"]


@pytest.mark.asyncio
async def test_reasoning_tool_can_run_again_after_concrete_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _DeepthinkWithConcreteBetweenProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    executed: list[str] = []

    async def fake_execute(name: str, params: dict[str, Any]) -> str:
        _ = params
        executed.append(name)
        return "ok"

    monkeypatch.setattr(loop.tools, "execute", fake_execute)

    final, tools_used = await loop._run_agent_loop(
        initial_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        model="k2-think-v2/LLM360/K2-Think-V2",
        allow_reasoning_tools=True,
        max_reasoning_tool_calls=3,
    )

    assert final == "done"
    assert executed.count("deepthink") == 2
    assert executed.count("write_file") == 1
    assert executed.count("exec") == 1
    assert tools_used == ["deepthink", "write_file", "deepthink", "exec"]
