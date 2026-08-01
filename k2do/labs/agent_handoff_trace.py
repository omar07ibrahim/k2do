"""Credential-free end-to-end trace lab for the K2DO agent handoff.

This laboratory drives the production message bus, router, ``AgentLoop``,
``DeepThinkEngine``, filesystem tools, and session manager with a strict
scripted provider.  The provider validates every private request in memory;
the public receipt retains only labels, counts, and SHA-256 digests.

Run it with::

    python -m k2do.labs.agent_handoff_trace
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from k2do.agent.deepthink import DEFAULT_THINKERS, JUDGE_SYSTEM_PROMPT
from k2do.agent.loop import AgentLoop
from k2do.agent.router import classify_query
from k2do.bus.events import InboundMessage
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from k2do.session.manager import SessionManager

_SCHEMA = "k2do.agent-handoff-trace/v1"
_MODEL = "offline/handoff-contract"
_CHANNEL = "lab"
_CHAT_ID = "handoff"
_SESSION_KEY = f"{_CHANNEL}:{_CHAT_ID}"
_COMPLEXITY_THRESHOLD = 0.6
_QUERY = (
    "Design, implement, and verify a resilient workspace handoff: create a result "
    "file, read it back, and explain the verified outcome."
)
_RELATIVE_PATH = "artifacts/handoff-proof.txt"
_ARTIFACT_CONTENT = "handoff-proof:v1\nstate=verified\n"
_FINAL_RESPONSE = "Workspace handoff verified."
_JUDGE_VERDICT = "Use workspace tools to create and verify the requested evidence artifact."
_JUDGE_RESPONSE = (
    "[JUDGE REASONING]\n"
    "The pragmatic plan has the clearest executable verification boundary.\n"
    "[FINAL ANSWER]\n"
    f"{_JUDGE_VERDICT}\n"
    "[SELECTED: Pragmatist]"
)
_THINKER_RESPONSES = {
    "Analyst": "Define the write, read-back, and persistence invariants before execution.",
    "Creative": "Use a compact receipt to separate private payloads from public proof.",
    "Pragmatist": "Write one bounded artifact, read it back, then persist the tool sequence.",
}
_EXPECTED_TOOL_NAMES = [
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "exec",
    "web_search",
    "web_fetch",
    "message",
    "spawn",
    "deepthink",
    "refine",
]
_CONTINUE_PROMPT = (
    "Use the tool results to continue execution. "
    "If the task is complete, return the final answer now. "
    "Do not output planning/reflection sections."
)
_HANDOFF_PROMPT = (
    "Now complete the original request end-to-end. "
    "Use tools when needed. "
    "Do not claim any command/file action unless it was actually executed."
)
_DURATION_RE = re.compile(r"(?<=, )\d+ms(?=\)\n)")
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_BUILTIN_SKILLS = Path(__file__).resolve().parents[1] / "skills"


class HandoffTraceError(RuntimeError):
    """Raised when an observed request or result violates the lab contract."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise HandoffTraceError(code)


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _replace_one(pattern: str, replacement: str, value: str, code: str) -> str:
    updated, count = re.subn(pattern, replacement, value, count=1, flags=re.MULTILINE)
    _require(count == 1, code)
    return updated


def _normalize_skill_summary(summary: str) -> str:
    prefix = "# Skills\n\nTo use a skill, read its SKILL.md with read_file.\n\n<skills>\n"
    _require(summary.startswith(prefix), "context_skill_summary_prefix_changed")
    _require(summary.endswith("\n</skills>"), "context_skill_summary_suffix_changed")
    body = summary[len(prefix) : -len("\n</skills>")]
    blocks = re.findall(r"  <skill .*?\n  </skill>", body, flags=re.DOTALL)
    _require(blocks and "\n".join(blocks) == body, "context_skill_blocks_changed")

    expected_names = {path.parent.name for path in _BUILTIN_SKILLS.glob("*/SKILL.md")}
    normalized_blocks: list[tuple[str, str]] = []
    for block in blocks:
        names = re.findall(r"<name>([^<]+)</name>", block)
        locations = re.findall(r"<location>([^<]+)</location>", block)
        _require(len(names) == 1, "context_skill_name_cardinality_changed")
        _require(len(locations) == 1, "context_skill_location_cardinality_changed")
        location = Path(locations[0])
        _require(location.name == "SKILL.md", "context_skill_location_shape_changed")
        _require(location.parent.name == names[0], "context_skill_location_name_changed")
        normalized = re.sub(
            r'available="(?:true|false)"',
            'available="<availability>"',
            block,
        )
        normalized = normalized.replace(str(_BUILTIN_SKILLS), "<builtin-skills>")
        _require(str(_BUILTIN_SKILLS) not in normalized, "context_skill_path_not_normalized")
        normalized_blocks.append((names[0], normalized))

    _require(
        {name for name, _ in normalized_blocks} == expected_names,
        "context_skill_set_changed",
    )
    ordered = "\n".join(block for _, block in sorted(normalized_blocks))
    return f"{prefix}{ordered}\n</skills>"


def _normalize_system_prompt(prompt: str, workspace: Path) -> str:
    """Validate dynamic context fields and return a stable contract form."""

    parts = prompt.split("\n\n---\n\n")
    _require(len(parts) == 3, "context_section_count_changed")
    identity, active_skills, skill_summary = parts
    _require(
        identity.startswith("# K2DO -- AI Agent with DeepThink\n"),
        "context_identity_changed",
    )
    _require(active_skills.startswith("# Active Skills\n"), "context_active_skills_changed")

    time_match = re.search(
        r"(?m)^## Current Time\n"
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \([^)]+\) \([^)]+\)$",
        identity,
    )
    _require(time_match is not None, "context_time_shape_changed")
    identity = _replace_one(
        r"^## Current Time\n[^\n]+$",
        "## Current Time\n<current-time>",
        identity,
        "context_time_cardinality_changed",
    )
    identity = _replace_one(
        r"^## Runtime\n[^\n]+$",
        "## Runtime\n<runtime>",
        identity,
        "context_runtime_cardinality_changed",
    )

    workspace_text = str(workspace.resolve())
    _require(workspace_text in identity, "context_workspace_missing")
    identity = identity.replace(workspace_text, "<workspace>")
    _require(workspace_text not in identity, "context_workspace_not_normalized")
    _require(
        "Path: <workspace>" in identity
        and "<workspace>/memory/MEMORY.md" in identity
        and "<workspace>/memory/HISTORY.md" in identity
        and "<workspace>/skills/{name}/SKILL.md" in identity,
        "context_workspace_contract_changed",
    )

    normalized_summary = _normalize_skill_summary(skill_summary)
    normalized = "\n\n---\n\n".join([identity, active_skills, normalized_summary])
    _require(workspace_text not in normalized, "context_private_workspace_leaked")
    return normalized


@dataclass(frozen=True)
class _ProviderEvent:
    ordinal: int
    kind: str
    subject: str


class _StrictHandoffProvider(LLMProvider):
    """Scripted provider that validates requests and never opens a network client."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(api_key=None, api_base=None)
        self.workspace = workspace.resolve()
        self.active_calls = 0
        self.peak_active_calls = 0
        self.execution_turn = 0
        self.tool_sequence: list[str] = []
        self._base_prompt_contract: str | None = None
        self._arrived_thinkers: set[str] = set()
        self._thinker_release = asyncio.Event()
        self._events: list[_ProviderEvent] = []
        self._ordinal = 0
        self._deepthink_projections: list[dict[str, Any]] = []
        self._execution_projections: list[dict[str, Any]] = []
        self._tool_catalog_sha256: str | None = None

    def get_default_model(self) -> str:
        return _MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _require(model == _MODEL, "provider_model_changed")
        _require(max_tokens == 4096, "provider_max_tokens_changed")
        if tools is None:
            return await self._chat_deepthink(messages, temperature)
        return await self._chat_execution(messages, tools, temperature)

    async def _chat_deepthink(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> LLMResponse:
        _require(len(messages) == 2, "deepthink_message_count_changed")
        _require(
            all(set(message) == {"role", "content"} for message in messages),
            "deepthink_message_shape_changed",
        )
        _require(
            [message["role"] for message in messages] == ["system", "user"],
            "deepthink_role_order_changed",
        )
        _require(
            all(isinstance(message["content"], str) for message in messages),
            "deepthink_content_type_changed",
        )

        if messages[0]["content"] == JUDGE_SYSTEM_PROMPT:
            return await self._chat_judge(messages, temperature)
        return await self._chat_thinker(messages, temperature)

    async def _chat_thinker(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> LLMResponse:
        system_prompt, separator, role_prompt = messages[0]["content"].rpartition(
            "\n\n## Your Role\n"
        )
        _require(bool(separator), "thinker_role_separator_missing")
        matches = [config for config in DEFAULT_THINKERS if config["role_prompt"] == role_prompt]
        _require(len(matches) == 1, "thinker_role_not_allowlisted")
        config = matches[0]
        actor = str(config["name"])
        _require(messages[1]["content"] == _QUERY, "thinker_query_changed")
        _require(temperature == config["temperature"], "thinker_temperature_changed")

        normalized_system = _normalize_system_prompt(system_prompt, self.workspace)
        if self._base_prompt_contract is None:
            self._base_prompt_contract = normalized_system
        _require(
            normalized_system == self._base_prompt_contract,
            "thinker_base_context_changed",
        )
        projection = [
            {"role": "system", "content_sha256": _sha256_text(normalized_system)},
            {"role": "role", "content_sha256": _sha256_text(role_prompt)},
            {"role": "user", "content_sha256": _sha256_text(_QUERY)},
        ]
        self._deepthink_projections.append(
            {
                "actor": actor.lower(),
                "messages_sha256": _sha256_json(projection),
                "phase": "thinker",
                "temperature": temperature,
            }
        )

        self._begin_call(f"thinker.{actor.lower()}")
        try:
            _require(actor not in self._arrived_thinkers, "thinker_called_twice")
            self._arrived_thinkers.add(actor)
            self._record("barrier_arrived", f"thinker.{actor.lower()}")
            if self._arrived_thinkers == set(_THINKER_RESPONSES):
                self._record("barrier_released", "thinkers.parallel")
                self._thinker_release.set()
            await self._thinker_release.wait()
            self._record("call_succeeded", f"thinker.{actor.lower()}")
            return LLMResponse(content=_THINKER_RESPONSES[actor])
        finally:
            self._end_call()

    async def _chat_judge(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> LLMResponse:
        _require(temperature == 0.3, "judge_temperature_changed")
        _require(self._thinker_release.is_set(), "judge_started_before_barrier_release")
        _require(self.active_calls == 0, "judge_started_before_thinkers_drained")
        normalized_prompt = _DURATION_RE.sub("<duration-ms>", messages[1]["content"])
        _require(
            normalized_prompt == self._expected_judge_prompt(),
            "judge_message_contract_changed",
        )
        projection = [
            {"role": "system", "content_sha256": _sha256_text(JUDGE_SYSTEM_PROMPT)},
            {"role": "user", "content_sha256": _sha256_text(normalized_prompt)},
        ]
        self._deepthink_projections.append(
            {
                "actor": "judge",
                "messages_sha256": _sha256_json(projection),
                "phase": "judge",
                "temperature": temperature,
            }
        )
        self._begin_call("judge")
        try:
            self._record("call_succeeded", "judge")
            return LLMResponse(content=_JUDGE_RESPONSE)
        finally:
            self._end_call()

    def _expected_judge_prompt(self) -> str:
        responses = ""
        for index, config in enumerate(DEFAULT_THINKERS, 1):
            name = str(config["name"])
            responses += (
                f"\n### Agent {index}: {name} "
                f"(temp={config['temperature']}, <duration-ms>)\n"
                f"**Response:**\n{_THINKER_RESPONSES[name]}\n"
            )
        return (
            f"## Original Question\n{_QUERY}\n\n"
            f"## Agent Responses\n{responses}\n\n"
            "## Your Task\n"
            "1. Evaluate each agent's response\n"
            "2. Synthesize the BEST possible final answer combining strengths from all\n"
            '3. Start your response with a brief "[JUDGE REASONING]" section explaining '
            "your evaluation\n"
            '4. Then provide the "[FINAL ANSWER]" \u2014 this is what the user will see\n'
            '5. End with "[SELECTED: AgentName]" indicating the primary contributor'
        )

    async def _chat_execution(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
    ) -> LLMResponse:
        _require(temperature == 0.7, "execution_temperature_changed")
        self._validate_tool_catalog(tools)
        self.execution_turn += 1
        _require(self.execution_turn <= 3, "execution_turn_count_exceeded")

        normalized_messages = deepcopy(messages)
        _require(
            normalized_messages
            and normalized_messages[0].get("role") == "system"
            and isinstance(normalized_messages[0].get("content"), str),
            "execution_system_message_changed",
        )
        session_suffix = f"\n\n## Current Session\nChannel: {_CHANNEL}\nChat ID: {_CHAT_ID}"
        raw_system = normalized_messages[0]["content"]
        _require(raw_system.endswith(session_suffix), "execution_session_context_changed")
        normalized_base = _normalize_system_prompt(
            raw_system[: -len(session_suffix)],
            self.workspace,
        )
        _require(
            normalized_base == self._base_prompt_contract,
            "execution_base_context_changed",
        )
        normalized_messages[0]["content"] = normalized_base + session_suffix
        _require(
            normalized_messages == self._expected_execution_messages(self.execution_turn),
            f"execution_turn_{self.execution_turn}_messages_changed",
        )
        self._execution_projections.append(
            {
                "messages_sha256": _sha256_json(normalized_messages),
                "turn": self.execution_turn,
            }
        )

        subject = f"execution.turn_{self.execution_turn}"
        self._begin_call(subject)
        try:
            if self.execution_turn == 1:
                self.tool_sequence.append("write_file")
                response = LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="handoff-write",
                            name="write_file",
                            arguments={
                                "path": _RELATIVE_PATH,
                                "content": _ARTIFACT_CONTENT,
                            },
                        )
                    ],
                )
            elif self.execution_turn == 2:
                self.tool_sequence.append("read_file")
                response = LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="handoff-read",
                            name="read_file",
                            arguments={"path": _RELATIVE_PATH},
                        )
                    ],
                )
            else:
                response = LLMResponse(content=_FINAL_RESPONSE)
            self._record("call_succeeded", subject)
            return response
        finally:
            self._end_call()

    def _expected_execution_messages(self, turn: int) -> list[dict[str, Any]]:
        _require(self._base_prompt_contract is not None, "base_context_not_observed")
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    self._base_prompt_contract
                    + f"\n\n## Current Session\nChannel: {_CHANNEL}\nChat ID: {_CHAT_ID}"
                ),
            },
            {"role": "user", "content": _QUERY},
            {
                "role": "assistant",
                "content": (
                    "DeepThink guidance for this task:\n"
                    f"{_JUDGE_VERDICT}\n\n"
                    "Use it as planning input."
                ),
            },
            {"role": "user", "content": _HANDOFF_PROMPT},
        ]
        if turn >= 2:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "handoff-write",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {
                                            "path": _RELATIVE_PATH,
                                            "content": _ARTIFACT_CONTENT,
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "handoff-write",
                        "name": "write_file",
                        "content": (
                            f"Successfully wrote {len(_ARTIFACT_CONTENT)} bytes to {_RELATIVE_PATH}"
                        ),
                    },
                    {"role": "user", "content": _CONTINUE_PROMPT},
                ]
            )
        if turn >= 3:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "handoff-read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": _RELATIVE_PATH}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "handoff-read",
                        "name": "read_file",
                        "content": _ARTIFACT_CONTENT,
                    },
                    {"role": "user", "content": _CONTINUE_PROMPT},
                ]
            )
        return messages

    def _validate_tool_catalog(self, tools: list[dict[str, Any]]) -> None:
        _require(isinstance(tools, list), "tool_catalog_not_list")
        _require(
            all(
                isinstance(item, dict)
                and set(item) == {"type", "function"}
                and item["type"] == "function"
                and isinstance(item["function"], dict)
                and set(item["function"]) == {"name", "description", "parameters"}
                for item in tools
            ),
            "tool_catalog_shape_changed",
        )
        names = [item["function"]["name"] for item in tools]
        _require(names == _EXPECTED_TOOL_NAMES, "tool_catalog_names_changed")
        digest = _sha256_json(tools)
        if self._tool_catalog_sha256 is None:
            self._tool_catalog_sha256 = digest
        _require(digest == self._tool_catalog_sha256, "tool_catalog_changed_between_turns")

    def _begin_call(self, subject: str) -> None:
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        self._record("call_started", subject)

    def _end_call(self) -> None:
        self.active_calls -= 1
        _require(self.active_calls >= 0, "provider_active_count_negative")

    def _record(self, kind: str, subject: str) -> None:
        self._ordinal += 1
        self._events.append(_ProviderEvent(self._ordinal, kind, subject))

    def _event_index(self, kind: str, subject: str) -> int:
        matches = [
            event.ordinal
            for event in self._events
            if event.kind == kind and event.subject == subject
        ]
        _require(len(matches) == 1, "provider_event_cardinality_changed")
        return matches[0]

    def verify_complete(self) -> None:
        _require(self.active_calls == 0, "provider_calls_not_drained")
        _require(self.peak_active_calls == 3, "provider_parallel_peak_changed")
        _require(
            self._arrived_thinkers == set(_THINKER_RESPONSES),
            "provider_thinker_set_changed",
        )
        _require(self.execution_turn == 3, "provider_execution_turns_changed")
        _require(
            self.tool_sequence == ["write_file", "read_file"],
            "provider_tool_sequence_changed",
        )
        judge_start = self._event_index("call_started", "judge")
        for actor in ("analyst", "creative", "pragmatist"):
            _require(
                self._event_index("call_succeeded", f"thinker.{actor}") < judge_start,
                "judge_gate_changed",
            )
        _require(
            self._event_index("call_succeeded", "judge")
            < self._event_index("call_started", "execution.turn_1"),
            "handoff_started_before_judge_terminal",
        )

    @property
    def tool_catalog_sha256(self) -> str:
        _require(self._tool_catalog_sha256 is not None, "tool_catalog_not_observed")
        return self._tool_catalog_sha256

    def deepthink_contract_sha256(self) -> str:
        ordered = sorted(
            self._deepthink_projections,
            key=lambda item: (item["phase"], item["actor"]),
        )
        _require(len(ordered) == 4, "deepthink_projection_count_changed")
        return _sha256_json(ordered)

    def execution_contract_sha256(self) -> str:
        _require(len(self._execution_projections) == 3, "execution_projection_count_changed")
        return _sha256_json(self._execution_projections)


@dataclass(frozen=True)
class _WorkflowObservation:
    artifact_bytes: int
    artifact_sha256: str
    bus_inbound_messages: int
    bus_outbound_messages: int
    deepthink_contract_sha256: str
    execution_contract_sha256: str
    final_result_sha256: str
    normalized_session_sha256: str
    peak_provider_calls: int
    route: str
    session_roles: list[str]
    session_tools_used: list[str]
    tool_catalog_sha256: str
    tool_sequence: list[str]


def _validate_persisted_session(workspace: Path) -> tuple[list[str], list[str], str]:
    session_files = sorted((workspace / "sessions").glob("*.jsonl"))
    _require(len(session_files) == 1, "session_file_count_changed")
    raw_lines = [
        json.loads(line)
        for line in session_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(len(raw_lines) == 3, "session_record_line_count_changed")
    metadata = raw_lines[0]
    _require(
        set(metadata) == {"_type", "created_at", "updated_at", "metadata", "last_consolidated"},
        "session_metadata_shape_changed",
    )
    _require(metadata["_type"] == "metadata", "session_metadata_type_changed")
    _require(metadata["metadata"] == {}, "session_metadata_values_changed")
    _require(metadata["last_consolidated"] == 0, "session_consolidation_state_changed")
    datetime.fromisoformat(metadata["created_at"])
    datetime.fromisoformat(metadata["updated_at"])

    fresh_session = SessionManager(workspace).get_or_create(_SESSION_KEY)
    _require(len(fresh_session.messages) == 2, "session_message_count_changed")
    roles = [message.get("role") for message in fresh_session.messages]
    _require(roles == ["user", "assistant"], "session_role_order_changed")
    _require(fresh_session.messages[0].get("content") == _QUERY, "session_query_changed")
    _require(
        fresh_session.messages[1].get("content") == _FINAL_RESPONSE,
        "session_response_changed",
    )
    tools_used = fresh_session.messages[1].get("tools_used")
    _require(
        tools_used == ["deepthink", "write_file", "read_file"],
        "session_tools_changed",
    )
    for message in fresh_session.messages:
        _require(isinstance(message.get("timestamp"), str), "session_timestamp_missing")
        datetime.fromisoformat(message["timestamp"])

    projection = {
        "key_sha256": _sha256_text(_SESSION_KEY),
        "last_consolidated": metadata["last_consolidated"],
        "messages": [
            {
                "content_sha256": _sha256_text(message["content"]),
                "role": message["role"],
                "timestamp": "validated_then_normalized",
                "tools_used": message.get("tools_used", []),
            }
            for message in fresh_session.messages
        ],
        "metadata": metadata["metadata"],
        "timestamps": "validated_then_normalized",
    }
    return roles, list(tools_used), _sha256_json(projection)


async def _execute_workflow(workspace: Path) -> _WorkflowObservation:
    """Execute the real workflow in an existing private workspace."""

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    provider = _StrictHandoffProvider(workspace)
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=_MODEL,
        fallback_model=_MODEL,
        max_iterations=6,
        brave_api_key="offline-fixture-not-a-credential",
        restrict_to_workspace=True,
        deepthink_enabled=True,
        complexity_threshold=_COMPLEXITY_THRESHOLD,
        deepthink_max_agents=3,
        deepthink_thinker_timeout_s=2.0,
        deepthink_judge_timeout_s=2.0,
    )

    _require(
        classify_query(_QUERY, _COMPLEXITY_THRESHOLD) == "deepthink",
        "fixture_router_contract_changed",
    )
    inbound = InboundMessage(
        channel=_CHANNEL,
        sender_id="fixture-user",
        chat_id=_CHAT_ID,
        content=_QUERY,
    )
    await bus.publish_inbound(inbound)
    observed_inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=3.0)
    outbound = await asyncio.wait_for(loop._process_message(observed_inbound), timeout=5.0)
    _require(outbound is not None, "agent_outbound_missing")
    await bus.publish_outbound(outbound)
    observed_outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)

    _require(loop._last_route == "deepthink", "agent_route_changed")
    _require(observed_outbound.channel == _CHANNEL, "outbound_channel_changed")
    _require(observed_outbound.chat_id == _CHAT_ID, "outbound_chat_changed")
    _require(observed_outbound.content == _FINAL_RESPONSE, "outbound_content_changed")
    _require(bus.inbound_size == 0 and bus.outbound_size == 0, "bus_queues_not_drained")

    artifact = (workspace / _RELATIVE_PATH).resolve()
    _require(workspace == artifact or workspace in artifact.parents, "artifact_escaped_workspace")
    _require(artifact.is_file(), "artifact_not_created")
    artifact_content = artifact.read_text(encoding="utf-8")
    _require(artifact_content == _ARTIFACT_CONTENT, "artifact_readback_changed")

    roles, session_tools, session_digest = _validate_persisted_session(workspace)
    provider.verify_complete()
    return _WorkflowObservation(
        artifact_bytes=len(artifact.read_bytes()),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        bus_inbound_messages=1,
        bus_outbound_messages=1,
        deepthink_contract_sha256=provider.deepthink_contract_sha256(),
        execution_contract_sha256=provider.execution_contract_sha256(),
        final_result_sha256=_sha256_text(observed_outbound.content),
        normalized_session_sha256=session_digest,
        peak_provider_calls=provider.peak_active_calls,
        route=loop._last_route,
        session_roles=roles,
        session_tools_used=session_tools,
        tool_catalog_sha256=provider.tool_catalog_sha256,
        tool_sequence=list(provider.tool_sequence),
    )


def _workflow_projection() -> dict[str, list[Any]]:
    nodes = [
        "message_bus.inbound",
        "router.deepthink",
        "thinkers.parallel",
        "judge.synthesis",
        "agent_loop.handoff",
        "tool.write_file",
        "tool.read_file",
        "session.persist",
        "message_bus.outbound",
    ]
    edge_specs = [(nodes[index], nodes[index + 1], "then") for index in range(len(nodes) - 1)]
    return {
        "nodes": nodes,
        "edges": [
            {"from": source, "relation": relation, "to": target}
            for source, target, relation in edge_specs
        ],
    }


def _build_receipt(observation: _WorkflowObservation) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "status": "verified",
        "evidence_boundary": {
            "external_credentials": "not_required",
            "model_payloads": "digests_labels_and_counts_only",
            "network_provider": "not_used",
            "network_tools": "not_invoked",
            "provider": "strict_scripted_fake",
            "workspace": "temporary_and_removed",
        },
        "production_surface": [
            "agent_loop.process_message",
            "context.build_messages",
            "deepthink.parallel_thinkers",
            "message_bus.round_trip",
            "router.classify_query",
            "session.persistence",
            "tools.read_file",
            "tools.write_file",
        ],
        "route": {
            "classification": observation.route,
            "complexity_threshold": _COMPLEXITY_THRESHOLD,
            "input_sha256": _sha256_text(_QUERY),
        },
        "deepthink": {
            "judge_gate": "after_all_thinkers_terminal",
            "provider_peak_calls": observation.peak_provider_calls,
            "request_contract_sha256": observation.deepthink_contract_sha256,
            "selected": "pragmatist",
            "thinker_count": 3,
        },
        "handoff": {
            "artifact": {
                "bytes": observation.artifact_bytes,
                "kind": "text_file",
                "relative_path": _RELATIVE_PATH,
                "sha256": observation.artifact_sha256,
            },
            "execution_provider_turns": 3,
            "final_result_sha256": observation.final_result_sha256,
            "message_contract_sha256": observation.execution_contract_sha256,
            "tool_catalog_sha256": observation.tool_catalog_sha256,
            "tool_sequence": observation.tool_sequence,
        },
        "session": {
            "message_roles": observation.session_roles,
            "normalized_record_sha256": observation.normalized_session_sha256,
            "persisted": True,
            "timestamp_fields": "validated_then_normalized",
            "tools_used": observation.session_tools_used,
        },
        "message_bus": {
            "inbound_messages": observation.bus_inbound_messages,
            "outbound_messages": observation.bus_outbound_messages,
            "queues": "drained",
        },
        "cleanup": {
            "active_provider_calls": 0,
            "temporary_workspace": "removed",
        },
        "workflow": _workflow_projection(),
    }


def _require_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    _require(set(value) == expected, code)


def _validate_public_receipt(receipt: dict[str, Any]) -> None:
    _require_keys(
        receipt,
        {
            "cleanup",
            "deepthink",
            "evidence_boundary",
            "handoff",
            "message_bus",
            "production_surface",
            "route",
            "schema",
            "session",
            "status",
            "workflow",
        },
        "public_root_keys_changed",
    )
    _require(receipt["schema"] == _SCHEMA, "public_schema_changed")
    _require(receipt["status"] == "verified", "public_status_changed")
    _require_keys(
        receipt["evidence_boundary"],
        {
            "external_credentials",
            "model_payloads",
            "network_provider",
            "network_tools",
            "provider",
            "workspace",
        },
        "public_boundary_keys_changed",
    )
    _require(
        receipt["evidence_boundary"]
        == {
            "external_credentials": "not_required",
            "model_payloads": "digests_labels_and_counts_only",
            "network_provider": "not_used",
            "network_tools": "not_invoked",
            "provider": "strict_scripted_fake",
            "workspace": "temporary_and_removed",
        },
        "public_boundary_values_changed",
    )
    _require_keys(
        receipt["route"],
        {"classification", "complexity_threshold", "input_sha256"},
        "public_route_keys_changed",
    )
    _require(
        receipt["route"]["classification"] == "deepthink"
        and receipt["route"]["complexity_threshold"] == _COMPLEXITY_THRESHOLD,
        "public_route_values_changed",
    )
    _require_keys(
        receipt["deepthink"],
        {
            "judge_gate",
            "provider_peak_calls",
            "request_contract_sha256",
            "selected",
            "thinker_count",
        },
        "public_deepthink_keys_changed",
    )
    _require(
        receipt["deepthink"]["judge_gate"] == "after_all_thinkers_terminal"
        and receipt["deepthink"]["provider_peak_calls"] == 3
        and receipt["deepthink"]["selected"] == "pragmatist"
        and receipt["deepthink"]["thinker_count"] == 3,
        "public_deepthink_values_changed",
    )
    _require_keys(
        receipt["handoff"],
        {
            "artifact",
            "execution_provider_turns",
            "final_result_sha256",
            "message_contract_sha256",
            "tool_catalog_sha256",
            "tool_sequence",
        },
        "public_handoff_keys_changed",
    )
    _require_keys(
        receipt["handoff"]["artifact"],
        {"bytes", "kind", "relative_path", "sha256"},
        "public_artifact_keys_changed",
    )
    _require(
        receipt["handoff"]["artifact"]["bytes"] == len(_ARTIFACT_CONTENT.encode())
        and receipt["handoff"]["artifact"]["kind"] == "text_file"
        and receipt["handoff"]["artifact"]["relative_path"] == _RELATIVE_PATH
        and receipt["handoff"]["execution_provider_turns"] == 3
        and receipt["handoff"]["tool_sequence"] == ["write_file", "read_file"],
        "public_handoff_values_changed",
    )
    _require_keys(
        receipt["session"],
        {
            "message_roles",
            "normalized_record_sha256",
            "persisted",
            "timestamp_fields",
            "tools_used",
        },
        "public_session_keys_changed",
    )
    _require(
        receipt["session"]["message_roles"] == ["user", "assistant"]
        and receipt["session"]["persisted"] is True
        and receipt["session"]["timestamp_fields"] == "validated_then_normalized"
        and receipt["session"]["tools_used"] == ["deepthink", "write_file", "read_file"],
        "public_session_values_changed",
    )
    _require(
        receipt["message_bus"]
        == {"inbound_messages": 1, "outbound_messages": 1, "queues": "drained"},
        "public_bus_values_changed",
    )
    _require(
        receipt["cleanup"] == {"active_provider_calls": 0, "temporary_workspace": "removed"},
        "public_cleanup_values_changed",
    )
    _require(
        receipt["workflow"] == _workflow_projection(),
        "public_workflow_changed",
    )
    _require(
        receipt["production_surface"]
        == [
            "agent_loop.process_message",
            "context.build_messages",
            "deepthink.parallel_thinkers",
            "message_bus.round_trip",
            "router.classify_query",
            "session.persistence",
            "tools.read_file",
            "tools.write_file",
        ],
        "public_production_surface_changed",
    )

    for key, value in _walk_items(receipt):
        if key.endswith("sha256"):
            _require(
                isinstance(value, str) and _HEX_DIGEST_RE.fullmatch(value) is not None,
                "public_digest_invalid",
            )

    encoded = _canonical_json(receipt)
    forbidden_exact = [
        _QUERY,
        _ARTIFACT_CONTENT,
        _FINAL_RESPONSE,
        _JUDGE_RESPONSE,
        _JUDGE_VERDICT,
        *_THINKER_RESPONSES.values(),
    ]
    _require(
        not any(value in encoded for value in forbidden_exact),
        "public_receipt_contains_private_fixture",
    )
    forbidden_patterns = (
        r"https?://",
        r'(?:^|[" ])/(?:etc|home|tmp|users|var)/',
        r"[a-z]:\\\\(?:users|windows|program files)\\\\",
        r"\\\\\\\\[^\\\"]+\\\\",
        r"api[_-]?key",
        r"bearer\s",
        r"\bghp_[a-z0-9_]+",
        r"\b(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?\b",
        r"\b(?:grpc|wss?)://",
        r"\b\d+(?:\.\d+)?ms\b",
    )
    _require(
        not any(re.search(pattern, encoded, re.IGNORECASE) for pattern in forbidden_patterns),
        "public_receipt_contains_forbidden_field",
    )
    _require(
        _canonical_json(json.loads(encoded)) == encoded,
        "public_receipt_not_canonical",
    )


def _walk_items(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)


async def run_handoff_lab() -> dict[str, Any]:
    """Run the credential-free workflow and return its safe receipt."""

    temporary = tempfile.TemporaryDirectory(prefix="k2do-handoff-")
    workspace = Path(temporary.name).resolve()
    try:
        observation = await _execute_workflow(workspace)
    finally:
        temporary.cleanup()
    _require(not workspace.exists(), "temporary_workspace_not_removed")
    receipt = _build_receipt(observation)
    _validate_public_receipt(receipt)
    return receipt


def render_receipt(receipt: dict[str, Any]) -> str:
    """Render a validated receipt with stable key ordering."""

    _validate_public_receipt(receipt)
    return _canonical_json(receipt, pretty=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary with non-sensitive, fail-closed errors."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        if args in (["-h"], ["--help"]):
            sys.stdout.write(
                "usage: python -m k2do.labs.agent_handoff_trace\n"
                "Run the credential-free K2DO end-to-end handoff trace laboratory.\n"
            )
            return 0
        sys.stderr.write(
            _canonical_json(
                {"failure": "invalid_invocation", "schema": _SCHEMA, "status": "failed"},
                pretty=True,
            )
        )
        return 2

    logger.disable("k2do.agent.loop")
    logger.disable("k2do.agent.deepthink")
    try:
        try:
            receipt = asyncio.run(run_handoff_lab())
        except Exception:
            sys.stderr.write(
                _canonical_json(
                    {"failure": "verification_failed", "schema": _SCHEMA, "status": "failed"},
                    pretty=True,
                )
            )
            return 1
    finally:
        logger.enable("k2do.agent.deepthink")
        logger.enable("k2do.agent.loop")
    sys.stdout.write(render_receipt(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
