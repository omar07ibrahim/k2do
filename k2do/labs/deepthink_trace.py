"""Deterministic, credential-free trace laboratory for K2DO DeepThink.

The laboratory executes the production router and :class:`DeepThinkEngine`
against a strict scripted provider. The provider validates full request
payloads in memory, while the public receipt contains only categorical labels,
counts, and SHA-256 digests.

Run it with::

    python -m k2do.labs.deepthink_trace

Wall-clock timeouts are used only as deadlock guards and to exercise the
production timeout path. They are deliberately excluded from the receipt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from k2do.agent.deepthink import (
    JUDGE_SYSTEM_PROMPT,
    DeepThinkEngine,
    DeepThinkResult,
)
from k2do.agent.router import classify_query
from k2do.providers.base import LLMProvider, LLMResponse

_SCHEMA = "k2do.deepthink-trace-lab/v1"
_PRIMARY_MODEL = "offline/primary"
_FALLBACK_MODEL = "offline/fallback"
_MODEL_LABELS = {
    _PRIMARY_MODEL: "primary",
    _FALLBACK_MODEL: "fallback",
}
_COMPLEX_QUERY = (
    "Design and compare two resilient event-processing architectures, then explain "
    "their failure trade-offs and verification plan."
)
_SIMPLE_QUERY = "hello"
_SYSTEM_PROMPT = "Synthetic offline verification context."
_LIVENESS_TIMEOUT_S = 3.0
_ENGINE_TIMEOUT_S = 1.0
_DURATION_RE = re.compile(r"(?<=, )\d+ms(?=\)\n)")


class TraceLabError(RuntimeError):
    """Raised when an observed trace violates the laboratory contract."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TraceLabError(code)


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


@dataclass(frozen=True)
class _ThinkerSpec:
    actor: str
    name: str
    temperature: float
    role_prompt: str

    def as_engine_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": _PRIMARY_MODEL,
            "temperature": self.temperature,
            "role_prompt": self.role_prompt,
        }


_THINKERS = (
    _ThinkerSpec(
        actor="analyst",
        name="Analyst",
        temperature=0.3,
        role_prompt="Synthetic analyst role contract.",
    ),
    _ThinkerSpec(
        actor="creative",
        name="Creative",
        temperature=0.9,
        role_prompt="Synthetic creative role contract.",
    ),
    _ThinkerSpec(
        actor="pragmatist",
        name="Pragmatist",
        temperature=0.5,
        role_prompt="Synthetic pragmatist role contract.",
    ),
    _ThinkerSpec(
        actor="observer",
        name="Observer",
        temperature=0.4,
        role_prompt="Synthetic observer role contract.",
    ),
)

_SYNTHESIS_BODIES = {
    "analyst": "synthetic-candidate-alpha",
    "creative": "synthetic-candidate-beta",
}
_JUDGE_SYNTHESIS = (
    "[JUDGE REASONING]\n"
    "synthetic-evaluation\n"
    "[FINAL ANSWER]\n"
    "synthetic-combined-verdict\n"
    "[SELECTED: Analyst]"
)
_JUDGE_REASONING = "synthetic-evaluation"
_JUDGE_VERDICT = "synthetic-combined-verdict"

_LONGEST_BODIES = {
    "analyst": "short-candidate",
    "creative": "longer-synthetic-candidate-with-coverage",
}


@dataclass(frozen=True)
class _JudgeEntry:
    name: str
    temperature: float
    response: str = ""
    error: str = ""


@dataclass(frozen=True)
class _Action:
    kind: Literal["success", "error", "block"]
    content: str = ""
    error_code: str = ""
    barrier: str = ""


@dataclass(frozen=True)
class _TraceEvent:
    ordinal: int
    kind: str
    subject: str


class _DeterministicBarrier:
    """An event barrier released by an exact, allowlisted participant set."""

    def __init__(
        self,
        name: str,
        expected: set[str],
        recorder: Callable[[str, str], None],
    ) -> None:
        self.name = name
        self.expected = frozenset(expected)
        self.arrived: set[str] = set()
        self.released = asyncio.Event()
        self._record = recorder

    async def arrive(self, call_id: str) -> None:
        _require(call_id in self.expected, "barrier_unexpected_participant")
        _require(call_id not in self.arrived, "barrier_duplicate_participant")
        self.arrived.add(call_id)
        self._record("barrier_arrived", call_id)
        if self.arrived == self.expected:
            self._record("barrier_released", f"barrier.{self.name}")
            self.released.set()
        await self.released.wait()

    @property
    def complete(self) -> bool:
        return self.released.is_set() and self.arrived == self.expected


class _ProgressProjection:
    """Maps production progress strings to a safe categorical event stream."""

    def __init__(self, thinker_names: Sequence[str]) -> None:
        self._thinker_names = set(thinker_names)
        self.events: list[tuple[str, str]] = []
        self.valid = True

    def __call__(self, name: str, status: str) -> None:
        actor = self._actor_label(name)
        category = self._status_label(name, status)
        if actor is None or category is None:
            self.valid = False
            return
        self.events.append((actor, category))

    def _actor_label(self, name: str) -> str | None:
        if name in self._thinker_names:
            return name.lower()
        if name == "DeepThink":
            return "orchestrator"
        if name == "Judge":
            return "judge"
        return None

    @staticmethod
    def _status_label(name: str, status: str) -> str | None:
        if name == "DeepThink" and status == "Spawning parallel agents...":
            return "spawn"
        if name == "DeepThink" and status.startswith("Complete (") and status.endswith("ms)"):
            return "complete"
        if name == "Judge" and status == "Evaluating all responses...":
            return "gate_open"
        if name == "Judge" and status == "Verdict ready":
            return "verdict_ready"
        if status == "Thinking...":
            return "started"
        if status.startswith("Done (") and status.endswith("ms)"):
            return "terminal_success"
        if status.startswith("Error: "):
            return "terminal_error"
        return None

    def index(self, actor: str, category: str) -> int:
        try:
            return self.events.index((actor, category))
        except ValueError as exc:
            raise TraceLabError("progress_event_missing") from exc


class _ScriptedProvider(LLMProvider):
    """Strict fake provider that never reads credentials or opens a network client."""

    def __init__(
        self,
        *,
        query: str,
        system_prompt: str,
        thinkers: Sequence[_ThinkerSpec],
        judge_entries: Sequence[_JudgeEntry],
        scripts: dict[tuple[str, str], _Action],
        barriers: dict[str, set[str]] | None = None,
    ) -> None:
        super().__init__(api_key=None, api_base=None)
        self._query = query
        self._system_prompt = system_prompt
        self._thinkers = {spec.actor: spec for spec in thinkers}
        self._judge_entries = tuple(judge_entries)
        self._scripts = dict(scripts)
        self._expected_call_ids = {f"{actor}.{model_label}" for actor, model_label in self._scripts}
        self._never = asyncio.Event()
        self._ordinal = 0
        self.events: list[_TraceEvent] = []
        self.request_projections: list[dict[str, Any]] = []
        self.active_calls = 0
        self.peak_active_calls = 0
        self.barriers = {
            name: _DeterministicBarrier(name, expected, self._record)
            for name, expected in (barriers or {}).items()
        }

    def get_default_model(self) -> str:
        return _PRIMARY_MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        actor, normalized_messages = self._validate_request(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_label = _MODEL_LABELS.get(model or "")
        _require(model_label is not None, "provider_model_not_allowlisted")
        call_id = f"{actor}.{model_label}"
        action = self._scripts.pop((actor, model_label), None)
        _require(action is not None, "provider_call_not_scripted")

        self.request_projections.append(
            {
                "actor": actor,
                "max_tokens": max_tokens,
                "message_contract_sha256": _sha256_json(normalized_messages),
                "model": model_label,
                "temperature": temperature,
            }
        )
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        self._record("call_started", call_id)

        try:
            if action.barrier:
                barrier = self.barriers.get(action.barrier)
                _require(barrier is not None, "provider_barrier_missing")
                await barrier.arrive(call_id)

            if action.kind == "success":
                self._record("call_succeeded", call_id)
                return LLMResponse(content=action.content)
            if action.kind == "error":
                raise RuntimeError(action.error_code)
            if action.kind == "block":
                await self._never.wait()
                raise TraceLabError("provider_block_released")
            raise TraceLabError("provider_action_unknown")
        except asyncio.CancelledError:
            self._record("call_cancelled", call_id)
            raise
        except Exception:
            self._record("call_failed", call_id)
            raise
        finally:
            self.active_calls -= 1
            _require(self.active_calls >= 0, "provider_active_count_negative")

    def _validate_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, list[dict[str, str]]]:
        _require(tools is None, "provider_tools_must_be_absent")
        _require(model in _MODEL_LABELS, "provider_model_not_allowlisted")
        _require(max_tokens == 4096, "provider_max_tokens_changed")
        _require(len(messages) == 2, "provider_message_count_changed")
        _require(
            all(set(message) == {"role", "content"} for message in messages),
            "provider_message_shape_changed",
        )
        _require(
            [message["role"] for message in messages] == ["system", "user"],
            "provider_role_order_changed",
        )
        _require(
            all(isinstance(message["content"], str) for message in messages),
            "provider_message_content_not_text",
        )

        if messages[0]["content"] == JUDGE_SYSTEM_PROMPT:
            _require(temperature == 0.3, "judge_temperature_changed")
            normalized_prompt = self._normalize_and_validate_judge_prompt(messages[1]["content"])
            return "judge", [
                {"role": "system", "content_sha256": _sha256_text(JUDGE_SYSTEM_PROMPT)},
                {"role": "user", "content_sha256": _sha256_text(normalized_prompt)},
            ]

        actor = self._identify_thinker(messages[0]["content"])
        spec = self._thinkers[actor]
        _require(temperature == spec.temperature, "thinker_temperature_changed")
        expected_messages = [
            {
                "role": "system",
                "content": f"{self._system_prompt}\n\n## Your Role\n{spec.role_prompt}",
            },
            {"role": "user", "content": self._query},
        ]
        _require(messages == expected_messages, "thinker_message_contract_changed")
        return actor, [
            {"role": message["role"], "content_sha256": _sha256_text(message["content"])}
            for message in messages
        ]

    def _identify_thinker(self, system_content: str) -> str:
        matches = [
            actor
            for actor, spec in self._thinkers.items()
            if system_content == f"{self._system_prompt}\n\n## Your Role\n{spec.role_prompt}"
        ]
        _require(len(matches) == 1, "thinker_identity_not_exact")
        return matches[0]

    def _normalize_and_validate_judge_prompt(self, prompt: str) -> str:
        normalized = _DURATION_RE.sub("<duration-ms>", prompt)
        expected = self._expected_judge_prompt()
        _require(normalized == expected, "judge_message_contract_changed")
        return normalized

    def _expected_judge_prompt(self) -> str:
        responses_text = ""
        for index, entry in enumerate(self._judge_entries, 1):
            if entry.error:
                responses_text += (
                    f"\n### Agent {index}: {entry.name} (FAILED)\nError: {entry.error}\n"
                )
            else:
                responses_text += (
                    f"\n### Agent {index}: {entry.name} "
                    f"(temp={entry.temperature}, <duration-ms>)\n"
                    f"**Response:**\n{entry.response}\n"
                )
        return (
            f"## Original Question\n{self._query}\n\n"
            f"## Agent Responses\n{responses_text}\n\n"
            "## Your Task\n"
            "1. Evaluate each agent's response\n"
            "2. Synthesize the BEST possible final answer combining strengths from all\n"
            '3. Start your response with a brief "[JUDGE REASONING]" section explaining '
            "your evaluation\n"
            '4. Then provide the "[FINAL ANSWER]" \u2014 this is what the user will see\n'
            '5. End with "[SELECTED: AgentName]" indicating the primary contributor'
        )

    def _record(self, kind: str, subject: str) -> None:
        self._ordinal += 1
        self.events.append(_TraceEvent(self._ordinal, kind, subject))

    def event_index(self, kind: str, subject: str) -> int:
        matches = [
            event.ordinal
            for event in self.events
            if event.kind == kind and event.subject == subject
        ]
        _require(len(matches) == 1, "provider_event_cardinality_changed")
        return matches[0]

    def assert_drained(self) -> None:
        _require(self.active_calls == 0, "provider_calls_not_drained")

    def assert_scripts_consumed(self) -> None:
        _require(not self._scripts, "provider_script_not_consumed")
        observed = {event.subject for event in self.events if event.kind == "call_started"}
        _require(observed == self._expected_call_ids, "provider_call_set_changed")

    def contract_digest(self) -> str:
        ordered = sorted(
            self.request_projections,
            key=lambda projection: (projection["actor"], projection["model"]),
        )
        return _sha256_json(ordered)


def _assert_event_order(
    provider: _ScriptedProvider,
    earlier: tuple[str, str],
    later: tuple[str, str],
) -> None:
    _require(
        provider.event_index(*earlier) < provider.event_index(*later),
        "provider_event_order_changed",
    )


def _assert_judge_gated(
    provider: _ScriptedProvider,
    terminal_calls: Sequence[tuple[str, str]],
) -> None:
    judge_start = provider.event_index("call_started", "judge.primary")
    for terminal_kind, call_id in terminal_calls:
        _require(
            provider.event_index(terminal_kind, call_id) < judge_start,
            "judge_started_before_thinkers_terminal",
        )


def _node(node_id: str, kind: str, outcome: str) -> dict[str, str]:
    return {"id": node_id, "kind": kind, "outcome": outcome}


def _edge(source: str, target: str, relation: str) -> dict[str, str]:
    return {"from": source, "relation": relation, "to": target}


def _require_deepthink_route(query: str) -> str:
    """Prove a scenario entered through the production router."""

    route = classify_query(query)
    _require(route == "deepthink", "scenario_router_did_not_select_deepthink")
    return route


async def _run_synthesis_scenario() -> dict[str, Any]:
    route = _require_deepthink_route(_COMPLEX_QUERY)
    active = _THINKERS[:3]
    timeout_error = f"Timed out after {_ENGINE_TIMEOUT_S:.0f}s"
    barrier_members = {
        "analyst.fallback",
        "creative.primary",
        "pragmatist.primary",
    }
    provider = _ScriptedProvider(
        query=_COMPLEX_QUERY,
        system_prompt=_SYSTEM_PROMPT,
        thinkers=active,
        judge_entries=(
            _JudgeEntry(
                name="Analyst",
                temperature=0.3,
                response=_SYNTHESIS_BODIES["analyst"],
            ),
            _JudgeEntry(
                name="Creative",
                temperature=0.9,
                response=_SYNTHESIS_BODIES["creative"],
            ),
            _JudgeEntry(name="Pragmatist", temperature=0.5, error=timeout_error),
        ),
        scripts={
            ("analyst", "primary"): _Action(
                kind="error",
                error_code="scripted_thinker_primary_error",
            ),
            ("analyst", "fallback"): _Action(
                kind="success",
                content=_SYNTHESIS_BODIES["analyst"],
                barrier="parallel",
            ),
            ("creative", "primary"): _Action(
                kind="success",
                content=_SYNTHESIS_BODIES["creative"],
                barrier="parallel",
            ),
            ("pragmatist", "primary"): _Action(
                kind="block",
                barrier="parallel",
            ),
            ("judge", "primary"): _Action(
                kind="success",
                content=_JUDGE_SYNTHESIS,
            ),
        },
        barriers={"parallel": barrier_members},
    )
    progress = _ProgressProjection([spec.name for spec in active])
    engine = DeepThinkEngine(
        chat_fn=provider.chat,
        model=_PRIMARY_MODEL,
        fallback_model=_FALLBACK_MODEL,
        thinkers=[spec.as_engine_config() for spec in _THINKERS],
        judge_model=_PRIMARY_MODEL,
        max_agents=3,
        thinker_timeout_s=_ENGINE_TIMEOUT_S,
        judge_timeout_s=_ENGINE_TIMEOUT_S,
    )

    try:
        result = await asyncio.wait_for(
            engine.think(
                query=_COMPLEX_QUERY,
                system_prompt=_SYSTEM_PROMPT,
                on_progress=progress,
            ),
            timeout=_LIVENESS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise TraceLabError("synthesis_scenario_liveness_failed") from exc

    _verify_synthesis_result(result)
    provider.assert_drained()
    provider.assert_scripts_consumed()
    _require(provider.barriers["parallel"].complete, "parallel_barrier_incomplete")
    _require(provider.peak_active_calls == 3, "parallel_peak_changed")
    _require(progress.valid, "progress_projection_invalid")
    _require(
        progress.index("judge", "gate_open")
        > max(
            progress.index("analyst", "terminal_success"),
            progress.index("creative", "terminal_success"),
            progress.index("pragmatist", "terminal_error"),
        ),
        "judge_progress_gate_changed",
    )

    _assert_event_order(
        provider,
        ("call_failed", "analyst.primary"),
        ("call_started", "analyst.fallback"),
    )
    _assert_event_order(
        provider,
        ("barrier_released", "barrier.parallel"),
        ("call_succeeded", "analyst.fallback"),
    )
    _assert_event_order(
        provider,
        ("barrier_released", "barrier.parallel"),
        ("call_succeeded", "creative.primary"),
    )
    _assert_judge_gated(
        provider,
        (
            ("call_succeeded", "analyst.fallback"),
            ("call_succeeded", "creative.primary"),
            ("call_cancelled", "pragmatist.primary"),
        ),
    )

    return {
        "id": "synthesis_with_thinker_fallback_and_timeout",
        "status": "verified",
        "call_dag": {
            "nodes": [
                _node("router", "router", route),
                _node("thinker.analyst.primary", "provider_call", "error"),
                _node("thinker.analyst.fallback", "provider_call", "success"),
                _node("thinker.creative.primary", "provider_call", "success"),
                _node("thinker.pragmatist.primary", "provider_call", "timeout"),
                _node("judge.primary", "provider_call", "synthesis"),
                _node("result", "orchestration", "success"),
            ],
            "edges": [
                _edge("router", "thinker.analyst.primary", "dispatch"),
                _edge("router", "thinker.creative.primary", "dispatch"),
                _edge("router", "thinker.pragmatist.primary", "dispatch"),
                _edge(
                    "thinker.analyst.primary",
                    "thinker.analyst.fallback",
                    "fallback_on_error",
                ),
                _edge("thinker.analyst.fallback", "judge.primary", "judge_gate"),
                _edge("thinker.creative.primary", "judge.primary", "judge_gate"),
                _edge("thinker.pragmatist.primary", "judge.primary", "judge_gate"),
                _edge("judge.primary", "result", "synthesize"),
            ],
        },
        "concurrency": {
            "configured_max_agents": 3,
            "provider_peak_calls": 3,
            "scripted_thinkers": 4,
            "unselected_thinkers": 1,
            "barrier_order": "all_arrived_before_release",
        },
        "fallback": {
            "actor": "analyst",
            "path": ["primary_error", "fallback_success"],
        },
        "timeout": {
            "actor": "pragmatist",
            "provider_call_cancelled": True,
            "result": "categorical_timeout",
        },
        "judge": {
            "gate": "after_all_thinkers_terminal",
            "mode": "parsed_synthesis",
            "selected": "analyst",
        },
        "cleanup": {
            "active_provider_calls": 0,
            "barriers_incomplete": 0,
            "root_task": "completed",
        },
        "request_contract_sha256": provider.contract_digest(),
        "result_sha256": _sha256_text(result.judge_verdict),
    }


def _verify_synthesis_result(result: DeepThinkResult) -> None:
    _require(result.query == _COMPLEX_QUERY, "synthesis_query_changed")
    _require(len(result.thinker_results) == 3, "synthesis_thinker_count_changed")
    by_name = {thinker.name: thinker for thinker in result.thinker_results}
    _require(
        list(by_name) == ["Analyst", "Creative", "Pragmatist"],
        "synthesis_thinker_order_changed",
    )
    analyst = by_name["Analyst"]
    creative = by_name["Creative"]
    pragmatist = by_name["Pragmatist"]
    _require(analyst.model == _FALLBACK_MODEL, "thinker_fallback_model_changed")
    _require(analyst.error is None, "thinker_fallback_result_changed")
    _require(
        analyst.response == _SYNTHESIS_BODIES["analyst"],
        "thinker_fallback_body_changed",
    )
    _require(creative.model == _PRIMARY_MODEL, "creative_model_changed")
    _require(creative.error is None, "creative_result_changed")
    _require(
        creative.response == _SYNTHESIS_BODIES["creative"],
        "creative_body_changed",
    )
    _require(pragmatist.model == _PRIMARY_MODEL, "timeout_model_changed")
    _require(pragmatist.response == "", "timeout_body_not_empty")
    _require(
        pragmatist.error == f"Timed out after {_ENGINE_TIMEOUT_S:.0f}s",
        "timeout_category_changed",
    )
    _require(result.judge_reasoning == _JUDGE_REASONING, "judge_reasoning_parse_changed")
    _require(result.judge_verdict == _JUDGE_VERDICT, "judge_verdict_parse_changed")
    _require(result.selected_thinker == "Analyst", "judge_selection_parse_changed")
    _require(
        all(thinker.duration_ms >= 0 for thinker in result.thinker_results),
        "thinker_clock_invalid",
    )
    _require(result.total_duration_ms >= 0, "orchestrator_clock_invalid")


async def _run_judge_fallback_scenario() -> dict[str, Any]:
    route = _require_deepthink_route(_COMPLEX_QUERY)
    active = _THINKERS[:2]
    barrier_members = {"analyst.primary", "creative.primary"}
    provider = _ScriptedProvider(
        query=_COMPLEX_QUERY,
        system_prompt=_SYSTEM_PROMPT,
        thinkers=active,
        judge_entries=(
            _JudgeEntry(
                name="Analyst",
                temperature=0.3,
                response=_LONGEST_BODIES["analyst"],
            ),
            _JudgeEntry(
                name="Creative",
                temperature=0.9,
                response=_LONGEST_BODIES["creative"],
            ),
        ),
        scripts={
            ("analyst", "primary"): _Action(
                kind="success",
                content=_LONGEST_BODIES["analyst"],
                barrier="parallel",
            ),
            ("creative", "primary"): _Action(
                kind="success",
                content=_LONGEST_BODIES["creative"],
                barrier="parallel",
            ),
            ("judge", "primary"): _Action(
                kind="error",
                error_code="scripted_judge_primary_error",
            ),
            ("judge", "fallback"): _Action(
                kind="error",
                error_code="scripted_judge_fallback_error",
            ),
        },
        barriers={"parallel": barrier_members},
    )
    progress = _ProgressProjection([spec.name for spec in active])
    engine = DeepThinkEngine(
        chat_fn=provider.chat,
        model=_PRIMARY_MODEL,
        fallback_model=_FALLBACK_MODEL,
        thinkers=[spec.as_engine_config() for spec in active],
        judge_model=_PRIMARY_MODEL,
        max_agents=2,
        thinker_timeout_s=_ENGINE_TIMEOUT_S,
        judge_timeout_s=_ENGINE_TIMEOUT_S,
    )

    try:
        result = await asyncio.wait_for(
            engine.think(
                query=_COMPLEX_QUERY,
                system_prompt=_SYSTEM_PROMPT,
                on_progress=progress,
            ),
            timeout=_LIVENESS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise TraceLabError("judge_fallback_scenario_liveness_failed") from exc

    _require(result.query == _COMPLEX_QUERY, "judge_fallback_query_changed")
    _require(len(result.thinker_results) == 2, "judge_fallback_thinker_count_changed")
    _require(
        result.judge_verdict == _LONGEST_BODIES["creative"],
        "judge_longest_fallback_changed",
    )
    _require(result.selected_thinker == "Creative", "judge_fallback_selection_changed")
    _require(
        result.judge_reasoning
        == "Judge error: scripted_judge_fallback_error. Using best individual response.",
        "judge_fallback_reason_changed",
    )
    provider.assert_drained()
    provider.assert_scripts_consumed()
    _require(provider.barriers["parallel"].complete, "judge_fallback_barrier_incomplete")
    _require(provider.peak_active_calls == 2, "judge_fallback_peak_changed")
    _require(progress.valid, "judge_fallback_progress_invalid")
    _assert_judge_gated(
        provider,
        (
            ("call_succeeded", "analyst.primary"),
            ("call_succeeded", "creative.primary"),
        ),
    )
    _assert_event_order(
        provider,
        ("call_failed", "judge.primary"),
        ("call_started", "judge.fallback"),
    )

    return {
        "id": "judge_failure_uses_longest_successful_thinker",
        "status": "verified",
        "call_dag": {
            "nodes": [
                _node("router", "router", route),
                _node("thinker.analyst.primary", "provider_call", "success"),
                _node("thinker.creative.primary", "provider_call", "success"),
                _node("judge.primary", "provider_call", "error"),
                _node("judge.fallback", "provider_call", "error"),
                _node("result", "orchestration", "fallback_success"),
            ],
            "edges": [
                _edge("router", "thinker.analyst.primary", "dispatch"),
                _edge("router", "thinker.creative.primary", "dispatch"),
                _edge("thinker.analyst.primary", "judge.primary", "judge_gate"),
                _edge("thinker.creative.primary", "judge.primary", "judge_gate"),
                _edge("judge.primary", "judge.fallback", "fallback_on_error"),
                _edge("judge.fallback", "result", "longest_successful_thinker"),
            ],
        },
        "concurrency": {
            "configured_max_agents": 2,
            "provider_peak_calls": 2,
            "barrier_order": "all_arrived_before_release",
        },
        "judge": {
            "gate": "after_all_thinkers_terminal",
            "model_path": ["primary_error", "fallback_error"],
            "mode": "longest_successful_thinker",
            "selected": "creative",
        },
        "cleanup": {
            "active_provider_calls": 0,
            "barriers_incomplete": 0,
            "root_task": "completed",
        },
        "request_contract_sha256": provider.contract_digest(),
        "result_sha256": _sha256_text(result.judge_verdict),
    }


async def _run_cancellation_scenario() -> dict[str, Any]:
    route = _require_deepthink_route(_COMPLEX_QUERY)
    active = _THINKERS[:3]
    barrier_members = {f"{spec.actor}.primary" for spec in active}
    provider = _ScriptedProvider(
        query=_COMPLEX_QUERY,
        system_prompt=_SYSTEM_PROMPT,
        thinkers=active,
        judge_entries=(),
        scripts={
            (spec.actor, "primary"): _Action(kind="block", barrier="started") for spec in active
        },
        barriers={"started": barrier_members},
    )
    progress = _ProgressProjection([spec.name for spec in active])
    engine = DeepThinkEngine(
        chat_fn=provider.chat,
        model=_PRIMARY_MODEL,
        fallback_model=_FALLBACK_MODEL,
        thinkers=[spec.as_engine_config() for spec in active],
        judge_model=_PRIMARY_MODEL,
        max_agents=3,
        thinker_timeout_s=_LIVENESS_TIMEOUT_S,
        judge_timeout_s=_ENGINE_TIMEOUT_S,
    )

    root_task = asyncio.create_task(
        engine.think(
            query=_COMPLEX_QUERY,
            system_prompt=_SYSTEM_PROMPT,
            on_progress=progress,
        )
    )
    try:
        await asyncio.wait_for(
            provider.barriers["started"].released.wait(),
            timeout=_LIVENESS_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        root_task.cancel()
        try:
            await root_task
        except asyncio.CancelledError:
            pass
        raise TraceLabError("cancellation_start_barrier_liveness_failed") from exc

    _require(provider.active_calls == 3, "cancellation_active_set_changed")
    root_task.cancel()
    cancelled = False
    try:
        await asyncio.wait_for(root_task, timeout=_LIVENESS_TIMEOUT_S)
    except asyncio.CancelledError:
        cancelled = True
    except asyncio.TimeoutError as exc:
        raise TraceLabError("cancellation_cleanup_liveness_failed") from exc

    _require(cancelled, "root_cancellation_not_propagated")
    _require(root_task.done() and root_task.cancelled(), "root_task_not_cancelled")
    provider.assert_drained()
    provider.assert_scripts_consumed()
    _require(provider.barriers["started"].complete, "cancellation_barrier_incomplete")
    _require(provider.peak_active_calls == 3, "cancellation_peak_changed")
    _require(progress.valid, "cancellation_progress_invalid")
    _require(
        not any(
            event.subject.startswith("judge.")
            for event in provider.events
            if event.kind == "call_started"
        ),
        "judge_started_after_cancellation",
    )
    for spec in active:
        call_id = f"{spec.actor}.primary"
        _assert_event_order(
            provider,
            ("barrier_released", "barrier.started"),
            ("call_cancelled", call_id),
        )

    return {
        "id": "caller_cancellation_drains_parallel_thinkers",
        "status": "verified",
        "call_dag": {
            "nodes": [
                _node("router", "router", route),
                _node("thinker.analyst.primary", "provider_call", "cancelled"),
                _node("thinker.creative.primary", "provider_call", "cancelled"),
                _node("thinker.pragmatist.primary", "provider_call", "cancelled"),
                _node("caller.cancel", "control", "propagated"),
                _node("cleanup", "orchestration", "drained"),
            ],
            "edges": [
                _edge("router", "thinker.analyst.primary", "dispatch"),
                _edge("router", "thinker.creative.primary", "dispatch"),
                _edge("router", "thinker.pragmatist.primary", "dispatch"),
                _edge("caller.cancel", "thinker.analyst.primary", "cancel"),
                _edge("caller.cancel", "thinker.creative.primary", "cancel"),
                _edge("caller.cancel", "thinker.pragmatist.primary", "cancel"),
                _edge("thinker.analyst.primary", "cleanup", "drain"),
                _edge("thinker.creative.primary", "cleanup", "drain"),
                _edge("thinker.pragmatist.primary", "cleanup", "drain"),
            ],
        },
        "cancellation": {
            "blocked_provider_calls": 3,
            "cancelled_provider_calls": 3,
            "judge_calls": 0,
            "propagation": "root_to_all_active_thinkers",
        },
        "concurrency": {
            "configured_max_agents": 3,
            "provider_peak_calls": 3,
            "barrier_order": "all_started_before_cancel",
        },
        "cleanup": {
            "active_provider_calls": 0,
            "barriers_incomplete": 0,
            "root_task": "cancelled",
        },
        "request_contract_sha256": provider.contract_digest(),
    }


def _verify_router() -> dict[str, Any]:
    complex_route = _require_deepthink_route(_COMPLEX_QUERY)
    simple_route = classify_query(_SIMPLE_QUERY)
    _require(complex_route == "deepthink", "complex_router_contract_changed")
    _require(simple_route == "simple", "simple_router_contract_changed")
    return {
        "cases": [
            {
                "input": "complex_architecture_fixture",
                "input_sha256": _sha256_text(_COMPLEX_QUERY),
                "route": complex_route,
            },
            {
                "input": "simple_greeting_fixture",
                "input_sha256": _sha256_text(_SIMPLE_QUERY),
                "route": simple_route,
            },
        ],
        "status": "verified",
    }


async def run_trace_lab() -> dict[str, Any]:
    """Run all deterministic scenarios and return a safe canonical projection."""

    router = _verify_router()
    scenarios = [
        await _run_synthesis_scenario(),
        await _run_judge_fallback_scenario(),
        await _run_cancellation_scenario(),
    ]

    receipt = {
        "schema": _SCHEMA,
        "status": "verified",
        "evidence_boundary": {
            "credentials": "not_read",
            "model_payloads": "digests_labels_and_counts_only",
            "network_provider": "not_used",
            "provider": "strict_scripted_fake",
            "timing": "liveness_guards_only",
        },
        "production_surface": [
            "deepthink.judge",
            "deepthink.model_fallback",
            "deepthink.parallel_thinkers",
            "router.classify_query",
        ],
        "router": router,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    _validate_public_receipt(receipt)
    return receipt


_PUBLIC_KEYS = frozenset(
    {
        "active_provider_calls",
        "actor",
        "barrier_order",
        "barriers_incomplete",
        "blocked_provider_calls",
        "call_dag",
        "cancellation",
        "cancelled_provider_calls",
        "cases",
        "cleanup",
        "concurrency",
        "configured_max_agents",
        "credentials",
        "edges",
        "evidence_boundary",
        "fallback",
        "from",
        "gate",
        "id",
        "input",
        "input_sha256",
        "judge",
        "judge_calls",
        "kind",
        "mode",
        "model_path",
        "model_payloads",
        "network_provider",
        "nodes",
        "outcome",
        "path",
        "production_surface",
        "propagation",
        "provider",
        "provider_call_cancelled",
        "provider_peak_calls",
        "relation",
        "request_contract_sha256",
        "result",
        "result_sha256",
        "root_task",
        "route",
        "router",
        "scenario_count",
        "scenarios",
        "schema",
        "scripted_thinkers",
        "selected",
        "status",
        "timeout",
        "timing",
        "to",
        "unselected_thinkers",
    }
)
_PUBLIC_STRINGS = frozenset(
    {
        "after_all_thinkers_terminal",
        "all_arrived_before_release",
        "all_started_before_cancel",
        "analyst",
        "cancel",
        "cancelled",
        "caller.cancel",
        "caller_cancellation_drains_parallel_thinkers",
        "categorical_timeout",
        "completed",
        "complex_architecture_fixture",
        "control",
        "creative",
        "cleanup",
        "deepthink",
        "deepthink.judge",
        "deepthink.model_fallback",
        "deepthink.parallel_thinkers",
        "digests_labels_and_counts_only",
        "dispatch",
        "drain",
        "drained",
        "error",
        "fallback_error",
        "fallback_on_error",
        "fallback_success",
        "judge.fallback",
        "judge.primary",
        "judge_failure_uses_longest_successful_thinker",
        "judge_gate",
        "k2do.deepthink-trace-lab/v1",
        "liveness_guards_only",
        "longest_successful_thinker",
        "not_read",
        "not_used",
        "orchestration",
        "parsed_synthesis",
        "pragmatist",
        "primary_error",
        "propagated",
        "provider_call",
        "result",
        "root_to_all_active_thinkers",
        "router",
        "router.classify_query",
        "simple",
        "simple_greeting_fixture",
        "strict_scripted_fake",
        "success",
        "synthesis",
        "synthesis_with_thinker_fallback_and_timeout",
        "synthesize",
        "thinker.analyst.fallback",
        "thinker.analyst.primary",
        "thinker.creative.primary",
        "thinker.pragmatist.primary",
        "timeout",
        "verified",
    }
)
_SCENARIO_KEYS = {
    "synthesis_with_thinker_fallback_and_timeout": frozenset(
        {
            "call_dag",
            "cleanup",
            "concurrency",
            "fallback",
            "id",
            "judge",
            "request_contract_sha256",
            "result_sha256",
            "status",
            "timeout",
        }
    ),
    "judge_failure_uses_longest_successful_thinker": frozenset(
        {
            "call_dag",
            "cleanup",
            "concurrency",
            "id",
            "judge",
            "request_contract_sha256",
            "result_sha256",
            "status",
        }
    ),
    "caller_cancellation_drains_parallel_thinkers": frozenset(
        {
            "call_dag",
            "cancellation",
            "cleanup",
            "concurrency",
            "id",
            "request_contract_sha256",
            "status",
        }
    ),
}


def _require_keys(value: Any, expected: frozenset[str], code: str) -> None:
    _require(isinstance(value, dict), f"{code}_not_object")
    _require(frozenset(value) == expected, f"{code}_keys_changed")


def _validate_public_structure(receipt: dict[str, Any]) -> None:
    _require_keys(
        receipt,
        frozenset(
            {
                "evidence_boundary",
                "production_surface",
                "router",
                "scenario_count",
                "scenarios",
                "schema",
                "status",
            }
        ),
        "public_root",
    )
    _require_keys(
        receipt["evidence_boundary"],
        frozenset(
            {
                "credentials",
                "model_payloads",
                "network_provider",
                "provider",
                "timing",
            }
        ),
        "public_boundary",
    )
    _require(
        receipt["production_surface"]
        == [
            "deepthink.judge",
            "deepthink.model_fallback",
            "deepthink.parallel_thinkers",
            "router.classify_query",
        ],
        "public_surface_changed",
    )
    _require_keys(receipt["router"], frozenset({"cases", "status"}), "public_router")
    cases = receipt["router"]["cases"]
    _require(isinstance(cases, list) and len(cases) == 2, "public_router_cases_changed")
    for case in cases:
        _require_keys(
            case,
            frozenset({"input", "input_sha256", "route"}),
            "public_router_case",
        )
    _require(
        [(case["input"], case["route"]) for case in cases]
        == [
            ("complex_architecture_fixture", "deepthink"),
            ("simple_greeting_fixture", "simple"),
        ],
        "public_router_projection_changed",
    )

    scenarios = receipt["scenarios"]
    _require(
        isinstance(scenarios, list) and receipt["scenario_count"] == len(scenarios) == 3,
        "public_scenario_count_changed",
    )
    _require(
        [scenario.get("id") for scenario in scenarios] == list(_SCENARIO_KEYS),
        "public_scenario_order_changed",
    )
    for scenario in scenarios:
        scenario_id = scenario["id"]
        _require_keys(scenario, _SCENARIO_KEYS[scenario_id], "public_scenario")
        _validate_public_scenario_structure(scenario)


def _validate_public_scenario_structure(scenario: dict[str, Any]) -> None:
    call_dag = scenario["call_dag"]
    _require_keys(call_dag, frozenset({"edges", "nodes"}), "public_dag")
    nodes = call_dag["nodes"]
    edges = call_dag["edges"]
    _require(isinstance(nodes, list) and nodes, "public_dag_nodes_changed")
    _require(isinstance(edges, list) and edges, "public_dag_edges_changed")
    for node in nodes:
        _require_keys(node, frozenset({"id", "kind", "outcome"}), "public_dag_node")
    node_ids = {node["id"] for node in nodes}
    _require(len(node_ids) == len(nodes), "public_dag_node_ids_not_unique")
    for edge in edges:
        _require_keys(edge, frozenset({"from", "relation", "to"}), "public_dag_edge")
        _require(
            edge["from"] in node_ids and edge["to"] in node_ids,
            "public_dag_edge_not_connected",
        )

    _require_keys(
        scenario["cleanup"],
        frozenset({"active_provider_calls", "barriers_incomplete", "root_task"}),
        "public_cleanup",
    )
    scenario_id = scenario["id"]
    concurrency_keys = {
        "synthesis_with_thinker_fallback_and_timeout": frozenset(
            {
                "barrier_order",
                "configured_max_agents",
                "provider_peak_calls",
                "scripted_thinkers",
                "unselected_thinkers",
            }
        ),
        "judge_failure_uses_longest_successful_thinker": frozenset(
            {"barrier_order", "configured_max_agents", "provider_peak_calls"}
        ),
        "caller_cancellation_drains_parallel_thinkers": frozenset(
            {"barrier_order", "configured_max_agents", "provider_peak_calls"}
        ),
    }
    _require_keys(
        scenario["concurrency"],
        concurrency_keys[scenario_id],
        "public_concurrency",
    )

    if scenario_id == "synthesis_with_thinker_fallback_and_timeout":
        _require_keys(scenario["fallback"], frozenset({"actor", "path"}), "public_fallback")
        _require_keys(
            scenario["timeout"],
            frozenset({"actor", "provider_call_cancelled", "result"}),
            "public_timeout",
        )
        _require_keys(
            scenario["judge"],
            frozenset({"gate", "mode", "selected"}),
            "public_judge",
        )
    elif scenario_id == "judge_failure_uses_longest_successful_thinker":
        _require_keys(
            scenario["judge"],
            frozenset({"gate", "mode", "model_path", "selected"}),
            "public_judge",
        )
    else:
        _require_keys(
            scenario["cancellation"],
            frozenset(
                {
                    "blocked_provider_calls",
                    "cancelled_provider_calls",
                    "judge_calls",
                    "propagation",
                }
            ),
            "public_cancellation",
        )


def _validate_public_values(value: Any, parent_key: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(key in _PUBLIC_KEYS, "public_key_not_allowlisted")
            _validate_public_values(child, key)
        return
    if isinstance(value, list):
        for child in value:
            _validate_public_values(child, parent_key)
        return
    if isinstance(value, str):
        if parent_key.endswith("_sha256"):
            _require(
                re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                "public_digest_invalid",
            )
        else:
            _require(value in _PUBLIC_STRINGS, "public_string_not_allowlisted")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        _require(0 <= value <= 4, "public_integer_not_allowlisted")
        return
    raise TraceLabError("public_value_type_not_allowlisted")


def _validate_public_receipt(receipt: dict[str, Any]) -> None:
    _validate_public_structure(receipt)
    _validate_public_values(receipt)
    encoded = _canonical_json(receipt)
    forbidden_exact = [
        _COMPLEX_QUERY,
        _SIMPLE_QUERY,
        _SYSTEM_PROMPT,
        _PRIMARY_MODEL,
        _FALLBACK_MODEL,
        _JUDGE_SYNTHESIS,
        _JUDGE_REASONING,
        _JUDGE_VERDICT,
        *_SYNTHESIS_BODIES.values(),
        *_LONGEST_BODIES.values(),
        *(spec.role_prompt for spec in _THINKERS),
    ]
    _require(
        not any(value in encoded for value in forbidden_exact),
        "public_receipt_contains_private_fixture",
    )
    forbidden_patterns = (
        r"https?://",
        r"(?:^|[\" ])/(?:etc|home|tmp|users|var)/",
        r"[a-z]:\\\\(?:users|windows|program files)\\\\",
        r"\\\\\\\\[^\\\"]+\\\\",
        r"api[_-]?key",
        r"bearer\s",
        r"\bghp_[a-z0-9_]+",
        r"\b(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?\b",
        r"\b(?:grpc|wss?)://",
        r"\b\d+(?:\.\d+)?ms\b",
        r"\b(?:latency|throughput|benchmark)\b",
    )
    _require(
        not any(re.search(pattern, encoded, re.IGNORECASE) for pattern in forbidden_patterns),
        "public_receipt_contains_forbidden_field",
    )
    _require(
        _canonical_json(json.loads(encoded)) == encoded,
        "public_receipt_not_canonical",
    )


def render_receipt(receipt: dict[str, Any]) -> str:
    """Render a receipt with stable key ordering and no terminal styling."""

    _validate_public_receipt(receipt)
    return _canonical_json(receipt, pretty=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with a fail-closed, non-sensitive error projection."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        if args in (["-h"], ["--help"]):
            sys.stdout.write(
                "usage: python -m k2do.labs.deepthink_trace\n"
                "Run the credential-free deterministic K2DO DeepThink trace laboratory.\n"
            )
            return 0
        failure = {
            "schema": _SCHEMA,
            "status": "failed",
            "failure": "invalid_invocation",
        }
        sys.stderr.write(_canonical_json(failure, pretty=True))
        return 2

    # This is the standalone CLI boundary: expected synthetic timeout/error
    # records must not contaminate its machine-readable stderr/stdout contract.
    # The reusable ``run_trace_lab`` API does not mutate Loguru state.
    logger.disable("k2do.agent.deepthink")
    try:
        try:
            receipt = asyncio.run(run_trace_lab())
        except Exception:
            failure = {
                "schema": _SCHEMA,
                "status": "failed",
                "failure": "verification_failed",
            }
            sys.stderr.write(_canonical_json(failure, pretty=True))
            return 1
    finally:
        logger.enable("k2do.agent.deepthink")
    sys.stdout.write(render_receipt(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
