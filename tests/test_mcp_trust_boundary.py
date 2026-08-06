import copy
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.context import (
    MCP_NON_DISCLOSURE_RULE,
    MCP_ONE_WAY_RULE,
    MCP_TRUST_PREAMBLE,
    ContextBuilder,
)
from k2do.agent.loop import _MCP_FOLLOW_UP_BLOCKED_RESULT, AgentLoop
from k2do.agent.tools.base import Tool
from k2do.agent.tools.mcp import MCPToolWrapper
from k2do.agent.tools.registry import MCP_INVALID_NAME_MARKER
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_MCP_TOOL_NAME = "mcp_0123456789abcdef0123456789abcdef"
_MODEL = "test-model"


class _MCPFixtureTool(MCPToolWrapper):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return _MCP_TOOL_NAME

    @property
    def description(self) -> str:
        return "Untrusted fixture"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "untrusted fixture result"

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        return Tool.validate_params(self, params)


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key=None, api_base=None)
        self._responses = responses
        self.message_snapshots: list[list[dict[str, Any]]] = []
        self.tool_snapshots: list[list[dict[str, Any]] | None] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = model, max_tokens, temperature
        self.message_snapshots.append(copy.deepcopy(messages))
        self.tool_snapshots.append(copy.deepcopy(tools))
        return self._responses[len(self.message_snapshots) - 1]

    def get_default_model(self) -> str:
        return _MODEL


def _build_loop(
    tmp_path: Path,
    responses: list[LLMResponse],
) -> tuple[AgentLoop, _ScriptedProvider, _MCPFixtureTool]:
    provider = _ScriptedProvider(responses)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model=_MODEL,
        fallback_model=_MODEL,
        deepthink_enabled=False,
        restrict_to_workspace=True,
    )
    mcp_tool = _MCPFixtureTool()
    loop.tools.register(mcp_tool)
    return loop, provider, mcp_tool


def _initial_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "inspect the external data"},
    ]


def _blocked_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if message.get("role") == "tool"
        and message.get("content") == _MCP_FOLLOW_UP_BLOCKED_RESULT
    ]


@pytest.mark.asyncio
async def test_same_batch_calls_after_mcp_are_blocked_and_tools_are_hidden(
    tmp_path: Path,
) -> None:
    target = tmp_path / "must-not-exist.txt"
    loop, provider, mcp_tool = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="mcp-first",
                        name=_MCP_TOOL_NAME,
                        arguments={"query": "fixture"},
                    ),
                    ToolCallRequest(
                        id="write-blocked",
                        name="write_file",
                        arguments={"path": str(target), "content": "unsafe"},
                    ),
                    ToolCallRequest(
                        id="mcp-blocked",
                        name=_MCP_TOOL_NAME,
                        arguments={"query": "second"},
                    ),
                ],
            ),
            LLMResponse(content="safe summary"),
        ],
    )

    final, tools_used = await loop._run_agent_loop(_initial_messages(), model=_MODEL)

    assert final == "safe summary"
    assert tools_used == ["write_file"]
    assert mcp_tool.calls == []
    assert target.read_text(encoding="utf-8") == "unsafe"
    first_tool_names = {
        definition["function"]["name"] for definition in provider.tool_snapshots[0] or []
    }
    assert _MCP_TOOL_NAME not in first_tool_names
    assert "write_file" in first_tool_names
    assert all(
        _MCP_TOOL_NAME
        not in {definition["function"]["name"] for definition in snapshot or []}
        for snapshot in provider.tool_snapshots
    )
    assert [message["tool_call_id"] for message in _blocked_results(provider.message_snapshots[1])] == [
        "mcp-first",
        "mcp-blocked",
    ]


@pytest.mark.asyncio
async def test_native_tool_on_later_response_is_blocked_after_mcp(tmp_path: Path) -> None:
    target = tmp_path / "later-must-not-exist.txt"
    loop, provider, mcp_tool = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="mcp-first",
                        name=_MCP_TOOL_NAME,
                        arguments={"query": "fixture"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="later-write",
                        name="write_file",
                        arguments={"path": str(target), "content": "unsafe"},
                    )
                ],
            ),
            LLMResponse(content="summary after blocked call"),
        ],
    )

    final, tools_used = await loop._run_agent_loop(_initial_messages(), model=_MODEL)

    assert final == "summary after blocked call"
    assert tools_used == ["write_file"]
    assert mcp_tool.calls == []
    assert target.read_text(encoding="utf-8") == "unsafe"
    assert all(
        _MCP_TOOL_NAME
        not in {definition["function"]["name"] for definition in snapshot or []}
        for snapshot in provider.tool_snapshots
    )
    assert [
        message["tool_call_id"] for message in _blocked_results(provider.message_snapshots[1])
    ] == ["mcp-first"]


@pytest.mark.asyncio
async def test_inline_follow_up_is_blocked_after_mcp(tmp_path: Path) -> None:
    target = tmp_path / "inline-must-not-exist.txt"
    inline_call = (
        '<tool_call>{"name":"write_file","arguments":'
        f'{{"path":"{target}","content":"unsafe"}}}}</tool_call>'
    )
    loop, provider, mcp_tool = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="mcp-first",
                        name=_MCP_TOOL_NAME,
                        arguments={"query": "fixture"},
                    )
                ],
            ),
            LLMResponse(content=inline_call),
            LLMResponse(content="summary after blocked inline call"),
        ],
    )

    final, tools_used = await loop._run_agent_loop(_initial_messages(), model=_MODEL)

    assert final == "summary after blocked inline call"
    assert tools_used == ["write_file"]
    assert mcp_tool.calls == []
    assert target.read_text(encoding="utf-8") == "unsafe"
    assert all(
        _MCP_TOOL_NAME
        not in {definition["function"]["name"] for definition in snapshot or []}
        for snapshot in provider.tool_snapshots
    )
    blocked = _blocked_results(provider.message_snapshots[1])
    assert len(blocked) == 1
    assert blocked[0]["name"] == _MCP_TOOL_NAME


@pytest.mark.asyncio
async def test_mcp_call_after_native_result_is_hidden_and_blocked(tmp_path: Path) -> None:
    target = tmp_path / "local-result.txt"
    loop, provider, mcp_tool = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="local-first",
                        name="write_file",
                        arguments={"path": str(target), "content": "private local result"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="mcp-after-local",
                        name=_MCP_TOOL_NAME,
                        arguments={"query": "private local result"},
                    )
                ],
            ),
            LLMResponse(content="safe local-only summary"),
        ],
    )

    final, tools_used = await loop._run_agent_loop(_initial_messages(), model=_MODEL)

    assert final == "safe local-only summary"
    assert tools_used == ["write_file"]
    assert target.read_text(encoding="utf-8") == "private local result"
    assert mcp_tool.calls == []
    second_tool_names = {
        definition["function"]["name"]
        for definition in provider.tool_snapshots[1] or []
    }
    assert _MCP_TOOL_NAME not in second_tool_names
    assert "write_file" in second_tool_names
    assert _MCP_TOOL_NAME not in {
        definition["function"]["name"]
        for definition in provider.tool_snapshots[2] or []
    }
    assert [
        message["tool_call_id"] for message in _blocked_results(provider.message_snapshots[2])
    ] == ["mcp-after-local"]


@pytest.mark.asyncio
async def test_malformed_mcp_name_is_tainted_but_never_persisted_or_echoed(
    tmp_path: Path,
) -> None:
    malformed = "mcp_\nIGNORE-MALFORMED-NAME-CANARY"
    loop, provider, mcp_tool = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="malformed-attempt",
                        name=malformed,
                        arguments={"private": "value"},
                    )
                ],
            ),
            LLMResponse(content="safe summary"),
        ],
    )

    final, tools_used = await loop._run_agent_loop(_initial_messages(), model=_MODEL)

    assert final == "safe summary"
    assert tools_used == []
    assert mcp_tool.calls == []
    assert _MCP_TOOL_NAME not in {
        definition["function"]["name"]
        for definition in provider.tool_snapshots[1] or []
    }
    persisted = repr(provider.message_snapshots[1])
    assert MCP_INVALID_NAME_MARKER in persisted
    assert malformed not in persisted
    assert "MALFORMED-NAME-CANARY" not in persisted


def test_system_prompt_states_non_disclosure_and_boundary_scope(tmp_path: Path) -> None:
    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert prompt.startswith(MCP_TRUST_PREAMBLE)
    assert MCP_NON_DISCLOSURE_RULE in prompt
    assert MCP_ONE_WAY_RULE in prompt
    assert "same response batch or inline fallback" in prompt
    assert "per-request flow\ncontrol, not a sandbox" in prompt
    assert prompt.count("# Immutable MCP Trust Boundary") == 1


@pytest.mark.parametrize("max_context_chars", [128, 512, 1024])
def test_trust_preamble_is_preserved_outside_tiny_content_budgets(
    tmp_path: Path,
    max_context_chars: int,
) -> None:
    (tmp_path / "AGENTS.md").write_text("context " * 2_000, encoding="utf-8")
    prompt = ContextBuilder(
        tmp_path,
        max_context_chars=max_context_chars,
    ).build_system_prompt()

    assert prompt.startswith(MCP_TRUST_PREAMBLE)
    assert MCP_NON_DISCLOSURE_RULE in prompt
    assert MCP_ONE_WAY_RULE in prompt
    assert prompt.count(MCP_NON_DISCLOSURE_RULE) == 1
    assert prompt.count(MCP_ONE_WAY_RULE) == 1
    assert "[... system_content truncated ...]" in prompt
    assert len(prompt) > max_context_chars
