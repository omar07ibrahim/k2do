from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.deepthink import DeepThinkResult
from k2do.agent.loop import AgentLoop
from k2do.agent.tools.base import Tool
from k2do.agent.tools.mcp import MCPConnectionReport, MCPServerOutcome, MCPToolWrapper
from k2do.agent.tools.registry import MCP_INVALID_NAME_MARKER
from k2do.bus.events import InboundMessage
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_MODEL = "offline/explicit-isolation"
_MCP_A = "mcp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_MCP_B = "mcp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_MCP_EXTRA = "mcp_cccccccccccccccccccccccccccccccc"
_MCP_UNKNOWN = "mcp_dddddddddddddddddddddddddddddddd"
_MCP_MALFORMED = "mcp_\nMALFORMED-NAME-CANARY"


@dataclass(frozen=True)
class _ProviderCall:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str | None


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse | BaseException]) -> None:
        super().__init__(api_key=None, api_base=None)
        self.responses = responses
        self.calls: list[_ProviderCall] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        del max_tokens, temperature
        self.calls.append(
            _ProviderCall(
                messages=copy.deepcopy(messages),
                tools=copy.deepcopy(tools),
                model=model,
            )
        )
        position = len(self.calls) - 1
        if position >= len(self.responses):
            return LLMResponse(content="UNEXPECTED-PROVIDER-CALL")
        response = self.responses[position]
        if isinstance(response, BaseException):
            raise response
        return response

    def get_default_model(self) -> str:
        return _MODEL


class _RecordingMCPTool(MCPToolWrapper):
    def __init__(
        self,
        name: str,
        *,
        result: str = "UNTRUSTED-MCP-RESULT",
        failure: BaseException | None = None,
    ) -> None:
        self._name = name
        self.result = result
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Untrusted external lookup fixture"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.result

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        return Tool.validate_params(self, params)


class _PoisonedLaterToolCall:
    """Any property access proves the explicit state machine inspected attempt two."""

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return super().__getattribute__(name)
        raise AssertionError("later MCP batch entry was inspected")


def _build_loop(
    tmp_path: Path,
    responses: list[LLMResponse | BaseException],
    *,
    deepthink_enabled: bool = False,
) -> tuple[AgentLoop, _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model=_MODEL,
        fallback_model=_MODEL,
        deepthink_enabled=deepthink_enabled,
        complexity_threshold=0.0,
        restrict_to_workspace=True,
    )
    return loop, provider


def _publish_snapshot(
    loop: AgentLoop,
    tools: tuple[_RecordingMCPTool, ...],
    *,
    extra_registered: tuple[_RecordingMCPTool, ...] = (),
) -> None:
    loop.tools.register_many([*tools, *extra_registered])
    names = tuple(tool.name for tool in tools)
    loop._mcp_tool_names = names
    loop._mcp_tools = tools
    loop._mcp_connected = True
    loop._mcp_report = MCPConnectionReport(
        outcomes=(
            MCPServerOutcome(
                position=0,
                status="connected",
                phase="ready",
                protocol_version="2025-11-25",
                tool_names=names,
            ),
        ),
        registered_tool_names=names,
        registered_tools=tools,
    )


def _tool_names(call: _ProviderCall) -> list[str]:
    return [definition["function"]["name"] for definition in call.tools or []]


def _assert_no_mcp_definitions(provider: _ScriptedProvider) -> None:
    assert provider.calls
    assert all(
        not any(name.startswith("mcp_") for name in _tool_names(call))
        for call in provider.calls
    )


def _explicit_message(
    content: str,
    *,
    media: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel="PRIVATE-CHANNEL-CANARY",
        sender_id="PRIVATE-SENDER-CANARY",
        chat_id="PRIVATE-CHAT-CANARY",
        content=content,
        media=media or [],
        metadata=metadata or {},
    )


def _tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call-1",
) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _inline_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>{payload}</tool_call>"


@pytest.mark.asyncio
async def test_chat_with_fallback_none_means_no_tools_not_registry_defaults(
    tmp_path: Path,
) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="ok")])
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    response = await loop._chat_with_fallback(
        messages=[{"role": "user", "content": "ordinary"}],
        model=_MODEL,
        tools=None,
    )

    assert response.content == "ok"
    assert provider.calls[0].tools == []


@pytest.mark.asyncio
@pytest.mark.parametrize("call_style", ["structured", "inline"])
@pytest.mark.parametrize("name", [_MCP_A, _MCP_MALFORMED])
async def test_normal_agent_loop_never_advertises_or_executes_forced_mcp_calls(
    tmp_path: Path,
    call_style: str,
    name: str,
) -> None:
    if call_style == "structured":
        first = LLMResponse(
            content="",
            tool_calls=[_tool_call(name, {"query": "PRIVATE-ARG-CANARY"})],
        )
    else:
        first = LLMResponse(content=_inline_tool_call(name, {"query": "PRIVATE-ARG-CANARY"}))
    loop, provider = _build_loop(
        tmp_path,
        [first, LLMResponse(content="safe ordinary answer")],
    )
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    await loop._run_agent_loop(
        [
            {"role": "system", "content": "ordinary-system"},
            {"role": "user", "content": "ordinary-user"},
        ],
        model=_MODEL,
    )

    assert mcp_tool.calls == []
    _assert_no_mcp_definitions(provider)


@pytest.mark.asyncio
async def test_simple_route_exposes_no_mcp_definitions(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="simple answer")])
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    assert await loop.process_direct("hello") == "simple answer"

    _assert_no_mcp_definitions(provider)
    assert mcp_tool.calls == []


@pytest.mark.asyncio
async def test_system_route_exposes_no_mcp_definitions(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="system answer")])
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    response = await loop._process_system_message(
        InboundMessage(
            channel="system",
            sender_id="timer",
            chat_id="cli:scheduled",
            content="ordinary background work",
        )
    )

    assert response is not None
    assert response.content == "system answer"
    _assert_no_mcp_definitions(provider)
    assert mcp_tool.calls == []


@pytest.mark.asyncio
async def test_deepthink_handoff_exposes_no_mcp_definitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [LLMResponse(content="deepthink handoff answer")],
        deepthink_enabled=True,
    )
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    async def fixed_deepthink(query: str, system_prompt: str) -> DeepThinkResult:
        del system_prompt
        return DeepThinkResult(query=query, judge_verdict="offline guidance")

    monkeypatch.setattr(loop, "_run_deepthink", fixed_deepthink)
    session = loop.sessions.get_or_create("cli:deepthink")
    response = await loop._process_deepthink(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="deepthink",
            content="compare two architectures",
        ),
        session,
    )

    assert response.content == "deepthink handoff answer"
    _assert_no_mcp_definitions(provider)
    assert mcp_tool.calls == []


@pytest.mark.asyncio
async def test_deepthink_execution_retry_exposes_no_mcp_definitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(content="plan only"),
            LLMResponse(content="still no execution"),
            LLMResponse(content="retry answer"),
        ],
        deepthink_enabled=True,
    )
    mcp_tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (mcp_tool,))

    async def fixed_deepthink(query: str, system_prompt: str) -> DeepThinkResult:
        del system_prompt
        return DeepThinkResult(query=query, judge_verdict="offline guidance")

    monkeypatch.setattr(loop, "_run_deepthink", fixed_deepthink)
    session = loop.sessions.get_or_create("cli:retry")
    response = await loop._process_deepthink(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="retry",
            content="create a verified artifact",
        ),
        session,
    )

    assert response.content == "retry answer"
    assert len(provider.calls) == 3
    _assert_no_mcp_definitions(provider)
    assert mcp_tool.calls == []


@pytest.mark.asyncio
async def test_explicit_first_envelope_is_fixed_minimal_and_snapshot_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Preserve This Query CASE"
    mcp_result = "UNTRUSTED-RESULT-CANARY"
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="FIRST-FREE-TEXT-CANARY",
                reasoning_content="FIRST-REASONING-CANARY",
                tool_calls=[_tool_call(_MCP_B, {"query": query})],
            ),
            LLMResponse(content="safe synthesis"),
        ],
    )
    tool_a = _RecordingMCPTool(_MCP_A)
    tool_b = _RecordingMCPTool(_MCP_B, result=mcp_result)
    extra = _RecordingMCPTool(_MCP_EXTRA)
    _publish_snapshot(loop, (tool_a, tool_b), extra_registered=(extra,))

    session = loop.sessions.get_or_create("PRIVATE-CHANNEL-CANARY:PRIVATE-CHAT-CANARY")
    session.add_message("user", "PRIVATE-HISTORY-CANARY")
    loop.sessions.save(session)
    before = loop.sessions._get_session_path(session.key).read_bytes()
    (tmp_path / "AGENTS.md").write_text("PRIVATE-BOOTSTRAP-CANARY", encoding="utf-8")
    (tmp_path / "memory" / "MEMORY.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "MEMORY.md").write_text(
        "PRIVATE-MEMORY-CANARY",
        encoding="utf-8",
    )

    def forbidden_context(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("explicit MCP mode touched ContextBuilder")

    def forbidden_session(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("explicit MCP mode touched SessionManager")

    monkeypatch.setattr(loop.context, "build_messages", forbidden_context)
    monkeypatch.setattr(loop.context, "build_system_prompt", forbidden_context)
    monkeypatch.setattr(loop.sessions, "get_or_create", forbidden_session)
    monkeypatch.setattr(loop.sessions, "save", forbidden_session)

    response = await loop._process_message(
        _explicit_message(
            f"  /McP\t {query}  ",
            metadata={"private": "PRIVATE-METADATA-CANARY"},
        )
    )

    assert response is not None
    assert response.content == "safe synthesis"
    assert loop._last_route == "mcp"
    assert loop._last_complexity == 0.0
    assert len(provider.calls) == 2
    first = provider.calls[0]
    assert len(first.messages) == 2
    assert first.messages[0].keys() == {"role", "content"}
    assert first.messages[0]["role"] == "system"
    assert first.messages[1] == {"role": "user", "content": query}
    assert _tool_names(first) == [_MCP_A, _MCP_B]
    first_wire = repr(first)
    for canary in (
        "PRIVATE-HISTORY-CANARY",
        "PRIVATE-BOOTSTRAP-CANARY",
        "PRIVATE-MEMORY-CANARY",
        "PRIVATE-CHANNEL-CANARY",
        "PRIVATE-SENDER-CANARY",
        "PRIVATE-CHAT-CANARY",
        "PRIVATE-METADATA-CANARY",
        str(tmp_path),
    ):
        assert canary not in first_wire
    assert tool_a.calls == []
    assert tool_b.calls == [{"query": query}]
    assert extra.calls == []
    assert provider.calls[1].tools == []
    synthesis_wire = repr(provider.calls[1].messages)
    assert "FIRST-FREE-TEXT-CANARY" not in synthesis_wire
    assert "FIRST-REASONING-CANARY" not in synthesis_wire
    assert loop.sessions._get_session_path(session.key).read_bytes() == before


@pytest.mark.asyncio
async def test_explicit_system_envelope_is_stable_across_requests(tmp_path: Path) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(tool_calls=[_tool_call(_MCP_A, {"query": "First"})], content=""),
            LLMResponse(content="first synthesis"),
            LLMResponse(tool_calls=[_tool_call(_MCP_A, {"query": "Second"})], content=""),
            LLMResponse(content="second synthesis"),
        ],
    )
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    first = await loop._process_message(_explicit_message("/mcp First"))
    second = await loop._process_message(_explicit_message("/MCP Second"))

    assert first is not None and first.content == "first synthesis"
    assert second is not None and second.content == "second synthesis"
    assert provider.calls[0].messages[0] == provider.calls[2].messages[0]
    fixed_system = provider.calls[0].messages[0]["content"]
    assert "First" not in fixed_system
    assert "Second" not in fixed_system
    assert str(tmp_path) not in fixed_system


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_snapshot", ["length", "name", "identity", "duplicate"])
async def test_inconsistent_explicit_snapshot_fails_closed_without_provider(
    tmp_path: Path,
    broken_snapshot: str,
) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    original = _RecordingMCPTool(_MCP_A)
    other = _RecordingMCPTool(_MCP_B)
    _publish_snapshot(loop, (original, other))

    if broken_snapshot == "length":
        loop._mcp_tools = (original,)
    elif broken_snapshot == "name":
        loop._mcp_tool_names = (_MCP_B, _MCP_A)
    elif broken_snapshot == "identity":
        loop.tools.unregister(_MCP_A)
        loop.tools.register(_RecordingMCPTool(_MCP_A))
    else:
        loop._mcp_tool_names = (_MCP_A, _MCP_A)
        loop._mcp_tools = (original, original)

    response = await loop._process_message(_explicit_message("/mcp safe query"))

    assert response is not None
    assert response.content.strip()
    assert provider.calls == []
    assert original.calls == []
    assert other.calls == []


@pytest.mark.asyncio
async def test_explicit_empty_catalog_fails_closed_without_provider(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])

    response = await loop._process_message(_explicit_message("/mcp safe query"))

    assert response is not None
    assert response.content.strip()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_explicit_all_server_failure_fails_closed_without_provider(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    loop._mcp_connected = True
    loop._mcp_report = MCPConnectionReport(
        outcomes=(
            MCPServerOutcome(
                position=0,
                status="failed",
                phase="connect",
                protocol_version=None,
                tool_names=(),
            ),
        ),
        registered_tool_names=(),
        registered_tools=(),
    )

    response = await loop._process_message(_explicit_message("/mcp safe query"))

    assert response is not None
    assert response.content.strip()
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_attempt",
    ["valid", "unknown", "malformed", "native", "invalid", "timeout"],
)
async def test_explicit_batch_consumes_exactly_one_attempt(
    tmp_path: Path,
    first_attempt: str,
) -> None:
    target = tmp_path / "must-not-exist.txt"
    failure: BaseException | None = None
    if first_attempt == "timeout":
        failure = asyncio.TimeoutError("PRIVATE-TIMEOUT-CANARY")
    tool_a = _RecordingMCPTool(_MCP_A, failure=failure)
    tool_b = _RecordingMCPTool(_MCP_B)

    first_name = {
        "valid": _MCP_A,
        "unknown": _MCP_UNKNOWN,
        "malformed": _MCP_MALFORMED,
        "native": "write_file",
        "invalid": _MCP_A,
        "timeout": _MCP_A,
    }[first_attempt]
    first_arguments: dict[str, Any]
    if first_attempt == "native":
        first_arguments = {"path": str(target), "content": "unsafe"}
    elif first_attempt == "invalid":
        first_arguments = {}
    else:
        first_arguments = {"query": "first"}

    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="FIRST-BATCH-FREE-TEXT-CANARY",
                reasoning_content="FIRST-BATCH-REASONING-CANARY",
                tool_calls=[
                    _tool_call(first_name, first_arguments, call_id="first-attempt"),
                    _tool_call(_MCP_B, {"query": "must stay blocked"}, call_id="second"),
                ],
            ),
            LLMResponse(content="safe synthesis"),
        ],
    )
    _publish_snapshot(loop, (tool_a, tool_b))

    response = await loop._process_message(_explicit_message("/mcp batch query"))

    assert response is not None
    assert response.content == "safe synthesis"
    assert len(provider.calls) == 2
    assert provider.calls[1].tools == []
    assert tool_b.calls == []
    assert not target.exists()
    if first_attempt in {"valid", "timeout"}:
        assert tool_a.calls == [{"query": "first"}]
    else:
        assert tool_a.calls == []
    synthesis_messages = provider.calls[1].messages
    synthesis_wire = repr(synthesis_messages)
    assert "FIRST-BATCH-FREE-TEXT-CANARY" not in synthesis_wire
    assert "FIRST-BATCH-REASONING-CANARY" not in synthesis_wire
    assert {message.get("tool_call_id") for message in synthesis_messages} == {
        None,
        "mcp-explicit-attempt",
    }
    assert "first-attempt" not in synthesis_wire
    assert '"second"' not in synthesis_wire
    if first_attempt == "malformed":
        assert _MCP_MALFORMED not in synthesis_wire
        assert "MALFORMED-NAME-CANARY" not in synthesis_wire


@pytest.mark.asyncio
async def test_explicit_never_inspects_later_batch_entries(tmp_path: Path) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    _tool_call(_MCP_A, {"query": "first"}),
                    _PoisonedLaterToolCall(),  # type: ignore[list-item]
                ],
            ),
            LLMResponse(content="safe synthesis"),
        ],
    )
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message("/mcp one attempt"))

    assert response is not None and response.content == "safe synthesis"
    assert tool.calls == [{"query": "first"}]
    synthesis_wire = repr(provider.calls[1].messages)
    assert "mcp-explicit-attempt" in synthesis_wire
    assert "call-1" not in synthesis_wire


@pytest.mark.asyncio
async def test_explicit_executes_owned_arguments_validated_before_identity_await(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _provider = _build_loop(tmp_path, [])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))
    snapshot = await loop._snapshot_explicit_mcp_catalog()
    assert snapshot is not None
    provider_arguments = {"query": "safe-before-await"}
    validation_finished = asyncio.Event()
    original_validate = tool.validate_params

    def observed_validation(arguments: dict[str, Any]) -> list[str]:
        errors = original_validate(arguments)
        validation_finished.set()
        return errors

    monkeypatch.setattr(tool, "validate_params", observed_validation)
    await loop._mcp_lifecycle_lock.acquire()
    attempt = asyncio.create_task(
        loop._execute_explicit_mcp_attempt(snapshot, _MCP_A, provider_arguments)
    )
    try:
        await validation_finished.wait()
        provider_arguments.clear()
        provider_arguments["unsafe"] = "MUTATED-AFTER-VALIDATION"
    finally:
        loop._mcp_lifecycle_lock.release()

    assert (await attempt).startswith("Untrusted external MCP data")
    assert tool.calls == [{"query": "safe-before-await"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("synthesis_style", ["structured", "inline"])
async def test_explicit_synthesis_tool_attempt_executes_nothing(
    tmp_path: Path,
    synthesis_style: str,
) -> None:
    target = tmp_path / "synthesis-must-not-exist.txt"
    tool_a = _RecordingMCPTool(_MCP_A)
    tool_b = _RecordingMCPTool(_MCP_B)
    if synthesis_style == "structured":
        malicious_synthesis = LLMResponse(
            content="SYNTHESIS-FREE-TEXT-CANARY",
            tool_calls=[
                _tool_call(
                    "write_file",
                    {"path": str(target), "content": "unsafe"},
                    call_id="native-synthesis",
                ),
                _tool_call(
                    _MCP_B,
                    {"query": "unsafe follow-up"},
                    call_id="mcp-synthesis",
                ),
            ],
        )
    else:
        malicious_synthesis = LLMResponse(
            content=_inline_tool_call(
                "write_file",
                {"path": str(target), "content": "unsafe"},
            )
        )

    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[_tool_call(_MCP_A, {"query": "lookup"})],
            ),
            malicious_synthesis,
        ],
    )
    _publish_snapshot(loop, (tool_a, tool_b))

    response = await loop._process_message(_explicit_message("/mcp lookup"))

    assert response is not None
    assert response.content.strip()
    assert "SYNTHESIS-FREE-TEXT-CANARY" not in response.content
    assert "<tool_call" not in response.content
    assert len(provider.calls) == 2
    assert provider.calls[1].tools == []
    assert tool_a.calls == [{"query": "lookup"}]
    assert tool_b.calls == []
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected_query"),
    [
        ("  /MCP\tMixed Query CASE  ", "Mixed Query CASE"),
        ("/mCp\nMixed Query CASE\n", "Mixed Query CASE"),
        ("\t/McP   Mixed Query CASE\t", "Mixed Query CASE"),
    ],
)
async def test_explicit_parser_is_case_insensitive_and_whitespace_aware(
    tmp_path: Path,
    content: str,
    expected_query: str,
) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[_tool_call(_MCP_A, {"query": expected_query})],
            ),
            LLMResponse(content="safe synthesis"),
        ],
    )
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message(content))

    assert response is not None
    assert response.content == "safe synthesis"
    assert provider.calls[0].messages[1] == {"role": "user", "content": expected_query}
    assert tool.calls == [{"query": expected_query}]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["/mcp", " /MCP   ", "\t/mCp\n\t"])
async def test_bare_explicit_command_is_rejected_without_provider(
    tmp_path: Path,
    content: str,
) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message(content))

    assert response is not None
    assert response.content.strip()
    assert provider.calls == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_explicit_command_with_media_is_rejected_without_provider(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(
        _explicit_message(
            "/mcp inspect attachment",
            media=["data:text/plain;base64,PRIVATE-MEDIA-CANARY"],
        )
    )

    assert response is not None
    assert response.content.strip()
    assert provider.calls == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_system_explicit_command_fails_before_context_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("system /mcp crossed the isolated routing guard")

    async def forbidden_system(*args: Any, **kwargs: Any) -> Any:
        return forbidden(*args, **kwargs)

    monkeypatch.setattr(loop, "_process_system_message", forbidden_system)
    monkeypatch.setattr(loop.context, "build_messages", forbidden)
    monkeypatch.setattr(loop.context, "build_system_prompt", forbidden)
    monkeypatch.setattr(loop.sessions, "get_or_create", forbidden)
    monkeypatch.setattr(loop.sessions, "save", forbidden)
    response = await loop._process_message(
        InboundMessage(
            channel="system",
            sender_id="PRIVATE-SYSTEM-SENDER-CANARY",
            chat_id="cli:PRIVATE-SYSTEM-CHAT-CANARY",
            content="/mcp PRIVATE-SYSTEM-QUERY-CANARY",
        )
    )

    assert response is not None
    assert response.content == "MCP retrieval is unavailable for system messages."
    assert "CANARY" not in response.content
    assert provider.calls == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_mcp_prefix_without_command_boundary_stays_in_normal_mode(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="ordinary answer")])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message("/mcpish ordinary text"))

    assert response is not None
    assert response.content == "ordinary answer"
    _assert_no_mcp_definitions(provider)
    assert tool.calls == []


@pytest.mark.asyncio
async def test_help_documents_explicit_mcp_syntax_without_provider(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])

    response = await loop._process_message(_explicit_message("/help"))

    assert response is not None
    assert "/mcp <request>" in response.content.lower()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_explicit_exchange_is_not_persisted_or_delayed_into_next_request(
    tmp_path: Path,
) -> None:
    query_canary = "EXPLICIT-QUERY-CANARY"
    result_canary = "EXPLICIT-RESULT-CANARY"
    final_canary = "EXPLICIT-FINAL-CANARY"
    normal_query = "ordinary follow-up"
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[_tool_call(_MCP_A, {"query": query_canary})],
            ),
            LLMResponse(content=final_canary),
            LLMResponse(content="ordinary answer"),
        ],
    )
    tool = _RecordingMCPTool(_MCP_A, result=result_canary)
    _publish_snapshot(loop, (tool,))
    session_key = "PRIVATE-CHANNEL-CANARY:PRIVATE-CHAT-CANARY"
    session_path = loop.sessions._get_session_path(session_key)

    explicit = await loop._process_message(
        _explicit_message(f"/mcp {query_canary}"),
        session_key=session_key,
    )

    assert explicit is not None and explicit.content == final_canary
    assert not session_path.exists()

    ordinary = await loop._process_message(
        InboundMessage(
            channel="PRIVATE-CHANNEL-CANARY",
            sender_id="user",
            chat_id="PRIVATE-CHAT-CANARY",
            content=normal_query,
        ),
        session_key=session_key,
    )

    assert ordinary is not None and ordinary.content == "ordinary answer"
    assert len(provider.calls) == 3
    normal_wire = repr(provider.calls[2].messages)
    for canary in (query_canary, result_canary, final_canary):
        assert canary not in normal_wire
    assert not any(
        name.startswith("mcp_") for name in _tool_names(provider.calls[2])
    )
    session = loop.sessions.get_or_create(session_key)
    assert [message["content"] for message in session.messages] == [
        normal_query,
        "ordinary answer",
    ]
    disk = session_path.read_text(encoding="utf-8")
    for canary in (query_canary, result_canary, final_canary):
        assert canary not in disk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unknown_name", "name_canary"),
    [
        (_MCP_UNKNOWN, _MCP_UNKNOWN),
        (_MCP_MALFORMED, "MALFORMED-NAME-CANARY"),
    ],
)
async def test_unknown_explicit_attempt_logs_and_synthesis_are_generic(
    tmp_path: Path,
    unknown_name: str,
    name_canary: str,
) -> None:
    argument_canary = "PRIVATE-UNKNOWN-ARGUMENT-CANARY"
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[
                    _tool_call(unknown_name, {"query": argument_canary}, call_id="unknown")
                ],
            ),
            LLMResponse(content="safe synthesis"),
        ],
    )
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))
    messages: list[str] = []
    loguru = __import__("loguru").logger
    sink = loguru.add(messages.append, format="{message}")
    try:
        response = await loop._process_message(_explicit_message("/mcp safe lookup"))
    finally:
        loguru.remove(sink)

    assert response is not None and response.content == "safe synthesis"
    assert tool.calls == []
    rendered = "".join(messages)
    assert name_canary not in rendered
    assert argument_canary not in rendered
    assert len(rendered) < 2_000
    synthesis_wire = repr(provider.calls[1].messages)
    assert name_canary not in synthesis_wire
    if unknown_name == _MCP_MALFORMED:
        assert MCP_INVALID_NAME_MARKER in synthesis_wire


@pytest.mark.asyncio
async def test_explicit_oversized_query_is_rejected_before_provider(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [LLMResponse(content="must not run")])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message("/mcp " + "q" * 16_385))

    assert response is not None and response.content.strip()
    assert provider.calls == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_explicit_provider_failure_is_generic_and_not_persisted(tmp_path: Path) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [RuntimeError("PRIVATE-PROVIDER-CANARY"), RuntimeError("PRIVATE-RETRY-CANARY")],
    )
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    response = await loop._process_message(_explicit_message("/mcp safe lookup"))

    assert response is not None and response.content.strip()
    assert "CANARY" not in response.content
    assert len(provider.calls) == 2
    assert tool.calls == []


@pytest.mark.asyncio
async def test_explicit_provider_cancellation_propagates(tmp_path: Path) -> None:
    loop, provider = _build_loop(tmp_path, [asyncio.CancelledError()])
    tool = _RecordingMCPTool(_MCP_A)
    _publish_snapshot(loop, (tool,))

    with pytest.raises(asyncio.CancelledError):
        await loop._process_message(_explicit_message("/mcp safe lookup"))

    assert len(provider.calls) == 1
    assert tool.calls == []


@pytest.mark.asyncio
async def test_explicit_tool_cancellation_propagates_without_synthesis(tmp_path: Path) -> None:
    loop, provider = _build_loop(
        tmp_path,
        [
            LLMResponse(
                content="",
                tool_calls=[_tool_call(_MCP_A, {"query": "lookup"})],
            )
        ],
    )
    tool = _RecordingMCPTool(_MCP_A, failure=asyncio.CancelledError())
    _publish_snapshot(loop, (tool,))

    with pytest.raises(asyncio.CancelledError):
        await loop._process_message(_explicit_message("/mcp lookup"))

    assert len(provider.calls) == 1
    assert tool.calls == [{"query": "lookup"}]
