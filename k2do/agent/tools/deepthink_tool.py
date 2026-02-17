"""
DeepThink Tool — the agent can call this when it faces a hard problem.

Instead of the user triggering DeepThink manually, the agent ITSELF
decides "this is complex, I need multiple perspectives" and calls
this tool. Multiple agents think in parallel, then a judge picks
the best answer.

Also includes a Refine tool for iterative improvement.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from k2do.agent.tools.base import Tool

if TYPE_CHECKING:
    from k2do.agent.deepthink import DeepThinkEngine
    from k2do.agent.refine import RefinementEngine


class DeepThinkTool(Tool):
    """Tool that spawns parallel agents for complex reasoning."""

    def __init__(self, engine: "DeepThinkEngine", system_prompt: str = ""):
        self._engine = engine
        self._system_prompt = system_prompt
        self._on_progress = None

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def set_progress_callback(self, cb):
        self._on_progress = cb

    @property
    def name(self) -> str:
        return "deepthink"

    @property
    def description(self) -> str:
        return (
            "Consult multiple AI agents in parallel for complex problems. "
            "Use this when you face a difficult question that benefits from "
            "multiple perspectives — design decisions, comparisons, debugging "
            "complex issues, architectural choices, or any problem where "
            "different viewpoints could lead to a better answer. "
            "Three specialists (Analyst, Creative, Pragmatist) think independently, "
            "then a Judge synthesizes the best response."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The specific question or problem to analyze from multiple perspectives. "
                        "Be detailed — include all context the parallel agents need."
                    ),
                },
            },
            "required": ["question"],
        }

    async def execute(self, **kwargs: Any) -> str:
        question = kwargs.get("question", "")
        if not question:
            return "Error: question is required"

        result = await self._engine.think(
            query=question,
            system_prompt=self._system_prompt,
            on_progress=self._on_progress,
        )

        # Keep the tool result compact to avoid context bloat in long sessions.
        status_line = []
        for r in result.thinker_results:
            if r.error:
                status_line.append(f"{r.name}:error")
            else:
                status_line.append(f"{r.name}:{r.duration_ms}ms")

        parts = [
            "## DeepThink Summary",
            f"Agents: {', '.join(status_line) if status_line else 'none'}",
        ]

        if result.judge_verdict:
            parts.append("### Final Verdict")
            parts.append(result.judge_verdict.strip())
            if result.selected_thinker:
                parts.append(f"Primary contributor: {result.selected_thinker}")
        else:
            best = max(result.thinker_results, key=lambda r: len(r.response)) if result.thinker_results else None
            if best and best.response:
                parts.append("### Fallback Answer")
                parts.append(best.response[:1200].strip())
            else:
                parts.append("### Fallback Answer")
                parts.append("DeepThink completed without a usable answer.")

        parts.append(f"Total time: {result.total_duration_ms}ms")
        return "\n\n".join(parts)


class RefineTool(Tool):
    """Tool that iteratively refines an answer through critic + improvement rounds."""

    def __init__(self, engine: "RefinementEngine", system_prompt: str = ""):
        self._engine = engine
        self._system_prompt = system_prompt
        self._on_round = None

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def set_round_callback(self, cb):
        self._on_round = cb

    @property
    def name(self) -> str:
        return "refine"

    @property
    def description(self) -> str:
        return (
            "Iteratively refine an answer through 3 rounds: "
            "1) Generate initial answer, "
            "2) Critic scores and identifies weaknesses, "
            "3) Refiner creates improved version addressing all issues. "
            "Use this when you want the highest quality answer possible — "
            "writing important code, drafting documents, solving tricky bugs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question or task to answer with maximum quality. "
                        "Include all relevant context."
                    ),
                },
            },
            "required": ["question"],
        }

    async def execute(self, **kwargs: Any) -> str:
        question = kwargs.get("question", "")
        if not question:
            return "Error: question is required"

        result = await self._engine.refine(
            query=question,
            system_prompt=self._system_prompt,
            on_round=self._on_round,
        )

        # Build report
        parts = ["## Refinement Results\n"]

        for r in result.rounds:
            label = {"initial": "Initial Answer", "critic": "Critic Analysis", "refiner": "Refined Answer"}.get(r.role, r.role)
            parts.append(f"### Round {r.round_num}: {label} ({r.duration_ms}ms)")

            if r.quality_markers:
                scores = ", ".join(f"{k}: {v}/10" for k, v in r.quality_markers.items())
                parts.append(f"**Scores:** {scores}")

            parts.append(r.content[:2000])
            parts.append("")

        if result.improvement_summary:
            parts.append(f"*{result.improvement_summary}*")

        parts.append(f"\n*Total time: {result.total_duration_ms}ms*")
        return "\n".join(parts)
