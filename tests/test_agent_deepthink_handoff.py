from pathlib import Path
from typing import Any

import pytest

from k2do.agent.deepthink import DeepThinkResult
from k2do.agent.loop import AgentLoop
from k2do.bus.events import InboundMessage
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _StubProvider(LLMProvider):
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
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_auto_deepthink_runs_execution_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=True,
        complexity_threshold=0.0,  # force deepthink route
    )

    async def fake_run_deepthink(query: str, system_prompt: str) -> DeepThinkResult:
        assert query == "build a wav file"
        assert isinstance(system_prompt, str)
        return DeepThinkResult(query=query, judge_verdict="Plan: write file, run script, verify output")

    captured: dict[str, Any] = {}

    async def fake_run_agent_loop(
        initial_messages: list[dict[str, Any]],
        model: str | None = None,
        require_concrete_tools: bool = False,
        allow_reasoning_tools: bool = True,
        max_reasoning_tool_calls: int = 3,
        reflect_mode: bool = False,
    ) -> tuple[str | None, list[str]]:
        captured["messages"] = initial_messages
        captured["model"] = model
        captured["require_concrete_tools"] = require_concrete_tools
        captured["allow_reasoning_tools"] = allow_reasoning_tools
        captured["max_reasoning_tool_calls"] = max_reasoning_tool_calls
        captured["reflect_mode"] = reflect_mode
        return "OK /tmp/music.wav 1234 4.0", ["write_file", "exec"]

    monkeypatch.setattr(loop, "_run_deepthink", fake_run_deepthink)
    monkeypatch.setattr(loop, "_run_agent_loop", fake_run_agent_loop)

    response = await loop.process_direct(
        "build a wav file",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
    )

    assert response == "OK /tmp/music.wav 1234 4.0"
    assert captured["model"] == "k2-think-v2/LLM360/K2-Think-V2"
    assert captured["require_concrete_tools"] is True
    assert captured["allow_reasoning_tools"] is True
    assert captured["max_reasoning_tool_calls"] == 2
    assert any(
        m.get("role") == "assistant" and "DeepThink guidance" in (m.get("content") or "")
        for m in captured["messages"]
    )

    session = loop.sessions.get_or_create("cli:test")
    assert session.messages[-1]["tools_used"] == ["deepthink", "write_file", "exec"]


@pytest.mark.asyncio
async def test_auto_deepthink_retries_when_no_concrete_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=True,
        complexity_threshold=0.0,
    )

    async def fake_run_deepthink(query: str, system_prompt: str) -> DeepThinkResult:
        _ = system_prompt
        return DeepThinkResult(query=query, judge_verdict="Plan only")

    calls = {"count": 0}

    async def fake_run_agent_loop(
        initial_messages: list[dict[str, Any]],
        model: str | None = None,
        require_concrete_tools: bool = False,
        allow_reasoning_tools: bool = True,
        max_reasoning_tool_calls: int = 3,
        reflect_mode: bool = False,
    ) -> tuple[str | None, list[str]]:
        _ = (
            initial_messages,
            model,
            require_concrete_tools,
            allow_reasoning_tools,
            max_reasoning_tool_calls,
            reflect_mode,
        )
        calls["count"] += 1
        if calls["count"] == 1:
            return "Planning response without execution", ["deepthink"]
        return "Execution done", ["write_file"]

    monkeypatch.setattr(loop, "_run_deepthink", fake_run_deepthink)
    monkeypatch.setattr(loop, "_run_agent_loop", fake_run_agent_loop)

    session = loop.sessions.get_or_create("cli:test_retry")
    message = InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="test_retry",
        content="сгенерируй музыку и запусти скрипт",
    )
    result = await loop._process_deepthink(message, session)

    assert result.content == "Execution done"
    assert calls["count"] == 2
    assert session.messages[-1]["tools_used"] == ["deepthink", "write_file"]
