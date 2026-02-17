import json
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse


class _InlineToolCallProvider(LLMProvider):
    def __init__(self, target_file: Path):
        super().__init__(api_key=None, api_base=None)
        self.target_file = target_file
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            payload = {
                "name": "write_file",
                "arguments": {
                    "path": str(self.target_file),
                    "content": "hello from inline tool call",
                },
            }
            return LLMResponse(content=f"<tool_call>\n{json.dumps(payload)}\n</tool_call>")
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


class _PlanThenInlineToolProvider(LLMProvider):
    def __init__(self, target_file: Path):
        super().__init__(api_key=None, api_base=None)
        self.target_file = target_file
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
            return LLMResponse(content="I planned the steps.")
        if self.calls == 2:
            payload = {
                "name": "write_file",
                "arguments": {
                    "path": str(self.target_file),
                    "content": "written after enforced action pass",
                },
            }
            return LLMResponse(content=f"<tool_call>\n{json.dumps(payload)}\n</tool_call>")
        return LLMResponse(content="done after execution")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


class _MalformedInlineToolProvider(LLMProvider):
    def __init__(self, target_file: Path):
        super().__init__(api_key=None, api_base=None)
        self.target_file = target_file
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
            payload = {
                "name": "write_file",
                "arguments": {
                    "path": str(self.target_file),
                    "content": "written via malformed wrapper",
                },
            }
            return LLMResponse(content=f"{json.dumps(payload)}\n</tool_call>")
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_inline_tool_call_fallback_executes_tool(tmp_path: Path) -> None:
    target_file = tmp_path / "inline.txt"
    provider = _InlineToolCallProvider(target_file)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
        restrict_to_workspace=True,
    )

    response = await loop.process_direct("create the file")

    assert response == "done"
    assert target_file.read_text(encoding="utf-8") == "hello from inline tool call"


@pytest.mark.asyncio
async def test_action_request_requires_concrete_tool_before_finalizing(tmp_path: Path) -> None:
    target_file = tmp_path / "enforced.txt"
    provider = _PlanThenInlineToolProvider(target_file)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
        restrict_to_workspace=True,
    )

    response = await loop.process_direct("создай файл enforced.txt")

    assert response == "done after execution"
    assert target_file.read_text(encoding="utf-8") == "written after enforced action pass"
    assert provider.calls >= 3


@pytest.mark.asyncio
async def test_malformed_inline_tool_call_is_still_parsed(tmp_path: Path) -> None:
    target_file = tmp_path / "malformed.txt"
    provider = _MalformedInlineToolProvider(target_file)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
        restrict_to_workspace=True,
    )

    response = await loop.process_direct("create malformed.txt")

    assert response == "done"
    assert target_file.read_text(encoding="utf-8") == "written via malformed wrapper"


def test_inline_tool_call_parser_ignores_embedded_json_snippets() -> None:
    snippet = 'Here is an example JSON: {"name":"exec","arguments":{"command":"echo hi"}}'
    assert AgentLoop._extract_inline_tool_call(snippet) is None
