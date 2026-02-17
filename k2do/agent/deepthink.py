"""
DeepThink Engine — Multi-Agent Parallel Reasoning with Judge.

When a complex query comes in, DeepThink:
1. Spawns N parallel "thinker" agents with different perspectives/temperatures
2. Each thinks independently and returns their answer
3. A Judge agent evaluates all answers and synthesizes the best response

This is the killer feature for the hackathon demo.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger


@dataclass
class ThinkerResult:
    """Result from one parallel thinker agent."""
    name: str
    model: str
    temperature: float
    role: str
    response: str
    reasoning: str | None = None
    duration_ms: int = 0
    error: str | None = None


@dataclass
class DeepThinkResult:
    """Complete DeepThink result with all thinker outputs and judge verdict."""
    query: str
    thinker_results: list[ThinkerResult] = field(default_factory=list)
    judge_verdict: str = ""
    judge_reasoning: str = ""
    selected_thinker: str = ""
    total_duration_ms: int = 0


# Default thinker configurations — different "perspectives"
DEFAULT_THINKERS = [
    {
        "name": "Analyst",
        "temperature": 0.3,
        "role_prompt": (
            "You are the Analyst. Think step-by-step with rigorous logic. "
            "Break down the problem systematically. Prioritize correctness and completeness. "
            "Show your reasoning chain clearly."
        ),
    },
    {
        "name": "Creative",
        "temperature": 0.9,
        "role_prompt": (
            "You are the Creative Thinker. Approach the problem from unexpected angles. "
            "Consider unconventional solutions. Think outside the box while staying practical. "
            "Propose innovative approaches."
        ),
    },
    {
        "name": "Pragmatist",
        "temperature": 0.5,
        "role_prompt": (
            "You are the Pragmatist. Focus on the most practical, efficient solution. "
            "Consider real-world constraints, edge cases, and implementation simplicity. "
            "Deliver a clean, actionable answer."
        ),
    },
]

JUDGE_SYSTEM_PROMPT = """You are the DeepThink Judge. You have received answers from multiple AI agents who independently analyzed the same question.

Your job:
1. Evaluate each agent's response for: correctness, completeness, clarity, and practicality
2. Identify the strengths and weaknesses of each
3. Synthesize the BEST possible answer, combining the strongest elements from all responses
4. Name which agent(s) contributed most to the final answer

Be concise but thorough. Your synthesized answer should be better than any individual response."""


class DeepThinkEngine:
    """
    Multi-agent parallel reasoning engine.

    Usage:
        engine = DeepThinkEngine(chat_fn=provider.chat)
        result = await engine.think(query, system_prompt, on_progress=callback)
    """

    def __init__(
        self,
        chat_fn: Callable[..., Awaitable[Any]],
        model: str = "",
        fallback_model: str = "",
        thinkers: list[dict] | None = None,
        judge_model: str | None = None,
        max_agents: int = 3,
        thinker_timeout_s: float = 90.0,
        judge_timeout_s: float = 60.0,
    ):
        self.chat_fn = chat_fn
        self.model = model
        self.fallback_model = fallback_model
        self.thinkers = thinkers or DEFAULT_THINKERS
        self.judge_model = judge_model or model
        self.max_agents = max(1, max_agents)
        self.thinker_timeout_s = max(0.5, thinker_timeout_s)
        self.judge_timeout_s = max(0.5, judge_timeout_s)

    async def think(
        self,
        query: str,
        system_prompt: str = "",
        on_progress: Callable[[str, str], None] | None = None,
    ) -> DeepThinkResult:
        """
        Run DeepThink: parallel agents + judge.

        Args:
            query: The user's question/task
            system_prompt: Base system prompt (identity, memory, etc.)
            on_progress: Callback(agent_name, status) for live UI updates
        """
        start = time.monotonic()
        result = DeepThinkResult(query=query)

        if on_progress:
            on_progress("DeepThink", "Spawning parallel agents...")

        # Phase 1: Run all thinkers in parallel
        active_thinkers = self.thinkers[:self.max_agents]
        tasks = []
        for thinker_cfg in active_thinkers:
            tasks.append(self._run_thinker(
                query=query,
                system_prompt=system_prompt,
                name=thinker_cfg["name"],
                model=thinker_cfg.get("model") or self.model,
                temperature=thinker_cfg.get("temperature", 0.7),
                role_prompt=thinker_cfg.get("role_prompt", ""),
                on_progress=on_progress,
            ))

        thinker_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in thinker_results:
            if isinstance(r, ThinkerResult):
                result.thinker_results.append(r)
                if on_progress:
                    status = f"Done ({r.duration_ms}ms)" if not r.error else f"Error: {r.error}"
                    on_progress(r.name, status)
            elif isinstance(r, Exception):
                logger.error(f"DeepThink thinker failed: {r}")

        # Phase 2: Judge evaluates all responses
        if len(result.thinker_results) > 0:
            if on_progress:
                on_progress("Judge", "Evaluating all responses...")

            judge_result = await self._run_judge(query, result.thinker_results, on_progress)
            result.judge_verdict = judge_result.get("verdict", "")
            result.judge_reasoning = judge_result.get("reasoning", "")
            result.selected_thinker = judge_result.get("selected", "")

        result.total_duration_ms = int((time.monotonic() - start) * 1000)

        if on_progress:
            on_progress("DeepThink", f"Complete ({result.total_duration_ms}ms)")

        return result

    async def _run_thinker(
        self,
        query: str,
        system_prompt: str,
        name: str,
        model: str,
        temperature: float,
        role_prompt: str,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> ThinkerResult:
        """Run a single thinker agent."""
        start = time.monotonic()

        if on_progress:
            on_progress(name, "Thinking...")

        # Build specialized system prompt for this thinker
        thinker_system = f"{system_prompt}\n\n## Your Role\n{role_prompt}" if system_prompt else role_prompt

        try:
            response, used_model = await asyncio.wait_for(
                self._chat_with_fallback(
                    messages=[
                        {"role": "system", "content": thinker_system},
                        {"role": "user", "content": query},
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=4096,
                ),
                timeout=self.thinker_timeout_s,
            )

            duration = int((time.monotonic() - start) * 1000)

            return ThinkerResult(
                name=name,
                model=used_model,
                temperature=temperature,
                role=role_prompt,
                response=response.content or "",
                reasoning=getattr(response, "reasoning_content", None),
                duration_ms=duration,
            )
        except asyncio.TimeoutError:
            duration = int((time.monotonic() - start) * 1000)
            err = f"Timed out after {self.thinker_timeout_s:.0f}s"
            logger.error(f"Thinker {name} failed: {err}")
            return ThinkerResult(
                name=name,
                model=model,
                temperature=temperature,
                role=role_prompt,
                response="",
                duration_ms=duration,
                error=err,
            )
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            logger.error(f"Thinker {name} failed: {e}")
            return ThinkerResult(
                name=name,
                model=model,
                temperature=temperature,
                role=role_prompt,
                response="",
                duration_ms=duration,
                error=str(e),
            )

    async def _run_judge(
        self,
        query: str,
        thinker_results: list[ThinkerResult],
        on_progress: Callable[[str, str], None] | None = None,
    ) -> dict[str, str]:
        """Judge evaluates all thinker responses and synthesizes the best answer."""

        # Build the evaluation prompt
        responses_text = ""
        for i, r in enumerate(thinker_results, 1):
            if r.error:
                responses_text += f"\n### Agent {i}: {r.name} (FAILED)\nError: {r.error}\n"
            else:
                reasoning_block = ""
                if r.reasoning:
                    reasoning_block = f"\n**Reasoning chain:**\n{r.reasoning}\n"
                responses_text += (
                    f"\n### Agent {i}: {r.name} (temp={r.temperature}, {r.duration_ms}ms)\n"
                    f"{reasoning_block}"
                    f"**Response:**\n{r.response}\n"
                )

        judge_prompt = f"""## Original Question
{query}

## Agent Responses
{responses_text}

## Your Task
1. Evaluate each agent's response
2. Synthesize the BEST possible final answer combining strengths from all
3. Start your response with a brief "[JUDGE REASONING]" section explaining your evaluation
4. Then provide the "[FINAL ANSWER]" — this is what the user will see
5. End with "[SELECTED: AgentName]" indicating the primary contributor"""

        try:
            response, _ = await asyncio.wait_for(
                self._chat_with_fallback(
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    model=self.judge_model,
                    temperature=0.3,
                    max_tokens=4096,
                ),
                timeout=self.judge_timeout_s,
            )

            text = response.content or ""

            # Parse judge output
            reasoning = ""
            verdict = text
            selected = ""

            if "[JUDGE REASONING]" in text and "[FINAL ANSWER]" in text:
                parts = text.split("[FINAL ANSWER]")
                reasoning = parts[0].replace("[JUDGE REASONING]", "").strip()
                verdict = parts[1] if len(parts) > 1 else text

            if "[SELECTED:" in verdict:
                sel_parts = verdict.split("[SELECTED:")
                verdict = sel_parts[0].strip()
                selected = sel_parts[1].replace("]", "").strip() if len(sel_parts) > 1 else ""

            if on_progress:
                on_progress("Judge", "Verdict ready")

            return {
                "verdict": verdict.strip(),
                "reasoning": reasoning.strip(),
                "selected": selected.strip(),
            }
        except asyncio.TimeoutError:
            logger.error(f"Judge failed: timed out after {self.judge_timeout_s:.0f}s")
            best = max(thinker_results, key=lambda r: len(r.response)) if thinker_results else None
            fallback = "DeepThink completed but judge timed out."
            return {
                "verdict": best.response if best and best.response.strip() else fallback,
                "reasoning": "Judge timed out. Using best individual response.",
                "selected": best.name if best else "",
            }
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            # Fallback: return the longest thinker response
            best = max(thinker_results, key=lambda r: len(r.response)) if thinker_results else None
            fallback = "DeepThink completed but judge failed."
            return {
                "verdict": best.response if best and best.response.strip() else fallback,
                "reasoning": f"Judge error: {e}. Using best individual response.",
                "selected": best.name if best else "",
            }

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[Any, str]:
        """Call model and retry once on fallback model if configured."""
        try:
            response = await self.chat_fn(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response, model
        except Exception:
            if self.fallback_model and self.fallback_model != model:
                response = await self.chat_fn(
                    messages=messages,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response, self.fallback_model
            raise
