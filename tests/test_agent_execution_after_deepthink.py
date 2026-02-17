from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _DeepthinkPlanThenExecuteProvider(LLMProvider):
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
                tool_calls=[
                    ToolCallRequest(
                        id="tc1",
                        name="deepthink",
                        arguments={"question": "plan this task"},
                    )
                ],
            )
        if self.calls == 2:
            return LLMResponse(content="Reflection on DeepThink output. Next steps: 1) do X 2) do Y")
        if self.calls == 3:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="tc2",
                        name="write_file",
                        arguments={"path": "demo.txt", "content": "ok"},
                    )
                ],
            )
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_deepthink_plan_response_forces_execution_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _DeepthinkPlanThenExecuteProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    async def fake_execute(name: str, params: dict[str, Any]) -> str:
        _ = name, params
        return "ok"

    monkeypatch.setattr(loop.tools, "execute", fake_execute)

    final, tools_used = await loop._run_agent_loop(
        initial_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        model="k2-think-v2/LLM360/K2-Think-V2",
        require_concrete_tools=False,
    )

    assert final == "done"
    assert tools_used == ["deepthink", "write_file"]
    # Must make an extra call after the planning-only response.
    assert provider.calls == 4


def test_action_hint_detection_handles_inflections() -> None:
    assert AgentLoop._looks_like_action_request("чувак это ты должен все сделать а не я")
    assert AgentLoop._looks_like_action_request("можешь сгенерировать python код и запустить скрипт")


@pytest.mark.asyncio
async def test_handoff_disables_reasoning_tools_in_chat_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _DeepthinkPlanThenExecuteProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    captured: dict[str, Any] = {}

    async def fake_chat_with_fallback(
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        _ = messages, model
        captured["tool_names"] = [d.get("function", {}).get("name") for d in (tools or [])]
        return LLMResponse(content="done")

    monkeypatch.setattr(loop, "_chat_with_fallback", fake_chat_with_fallback)

    final, tools_used = await loop._run_agent_loop(
        initial_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        model="k2-think-v2/LLM360/K2-Think-V2",
        allow_reasoning_tools=False,
    )

    assert final == "done"
    assert tools_used == []
    assert "deepthink" not in captured["tool_names"]
    assert "refine" not in captured["tool_names"]
