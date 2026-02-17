"""
Multi-Round Refinement Engine.

3-round process that visibly improves answers:
  Round 1: K2-Think generates initial response
  Round 2: Critic agent analyzes weaknesses, suggests improvements
  Round 3: Refiner synthesizes the best final answer

Each round is observable via callbacks for live dashboard display.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger


@dataclass
class RoundResult:
    """Result from one refinement round."""
    round_num: int
    role: str          # "initial", "critic", "refiner"
    content: str = ""
    reasoning: str | None = None
    duration_ms: int = 0
    quality_markers: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Complete refinement result across all rounds."""
    query: str
    rounds: list[RoundResult] = field(default_factory=list)
    total_duration_ms: int = 0
    improvement_summary: str = ""


CRITIC_SYSTEM = """You are a rigorous AI Critic. You receive an AI-generated answer and must:

1. **Score** the answer (1-10) on: accuracy, completeness, clarity, practicality
2. **Identify weaknesses**: What's missing? What's wrong? What could be better?
3. **Suggest improvements**: Specific, actionable suggestions

Format your response as:
## Scores
- Accuracy: X/10
- Completeness: X/10
- Clarity: X/10
- Practicality: X/10
- Overall: X/10

## Weaknesses
[List specific weaknesses]

## Improvements
[List specific suggestions for the refiner]"""

REFINER_SYSTEM = """You are an AI Refiner. You receive:
1. The original question
2. An initial answer
3. A critic's analysis with scores and improvement suggestions

Your job: Create a SIGNIFICANTLY IMPROVED answer that:
- Fixes all identified weaknesses
- Incorporates all improvement suggestions
- Maintains the strengths of the original
- Is clearer and more complete

Start with a brief note: "[REFINED: addressed X weaknesses, +Y improvements]"
Then provide the improved answer."""


class RefinementEngine:
    """
    Multi-round answer refinement.

    Usage:
        engine = RefinementEngine(chat_fn=provider.chat, model="...")
        result = await engine.refine(query, on_round=callback)
    """

    def __init__(
        self,
        chat_fn: Callable[..., Awaitable[Any]],
        model: str = "",
        critic_model: str | None = None,
        refiner_model: str | None = None,
    ):
        self.chat_fn = chat_fn
        self.model = model
        self.critic_model = critic_model or model
        self.refiner_model = refiner_model or model

    async def refine(
        self,
        query: str,
        system_prompt: str = "",
        on_round: Callable[[int, str, str], None] | None = None,
    ) -> RefinementResult:
        """
        Run 3-round refinement.

        Args:
            query: User's question
            system_prompt: Base system prompt
            on_round: Callback(round_num, role, status) for live updates
        """
        start = time.monotonic()
        result = RefinementResult(query=query)

        # ── Round 1: Initial Response ──────────────────────────
        if on_round:
            on_round(1, "K2-Think", "Generating initial response...")

        r1_start = time.monotonic()
        try:
            r1_resp = await self.chat_fn(
                messages=[
                    {"role": "system", "content": system_prompt or "You are a helpful AI assistant. Be thorough and detailed."},
                    {"role": "user", "content": query},
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=4096,
            )
            r1_content = r1_resp.content or ""
            r1_reasoning = getattr(r1_resp, "reasoning_content", None)
        except Exception as e:
            logger.error(f"Round 1 failed: {e}")
            r1_content = f"Error: {e}"
            r1_reasoning = None

        r1 = RoundResult(
            round_num=1, role="initial",
            content=r1_content, reasoning=r1_reasoning,
            duration_ms=int((time.monotonic() - r1_start) * 1000),
        )
        result.rounds.append(r1)

        if on_round:
            on_round(1, "K2-Think", f"Done ({r1.duration_ms}ms, {len(r1_content)} chars)")

        # ── Round 2: Critic ────────────────────────────────────
        if on_round:
            on_round(2, "Critic", "Analyzing response quality...")

        r2_start = time.monotonic()
        critic_prompt = f"""## Original Question
{query}

## AI Response to Critique
{r1_content}

Analyze this response thoroughly. Be constructive but honest."""

        try:
            r2_resp = await self.chat_fn(
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": critic_prompt},
                ],
                model=self.critic_model,
                temperature=0.3,
                max_tokens=2048,
            )
            r2_content = r2_resp.content or ""
        except Exception as e:
            logger.error(f"Round 2 (critic) failed: {e}")
            r2_content = f"Critic error: {e}"

        # Parse scores from critic output
        quality = self._parse_scores(r2_content)

        r2 = RoundResult(
            round_num=2, role="critic",
            content=r2_content,
            duration_ms=int((time.monotonic() - r2_start) * 1000),
            quality_markers=quality,
        )
        result.rounds.append(r2)

        if on_round:
            overall = quality.get("overall", "?")
            on_round(2, "Critic", f"Done ({r2.duration_ms}ms, score: {overall}/10)")

        # ── Round 3: Refiner ───────────────────────────────────
        if on_round:
            on_round(3, "Refiner", "Creating improved response...")

        r3_start = time.monotonic()
        refiner_prompt = f"""## Original Question
{query}

## Initial Response
{r1_content}

## Critic's Analysis
{r2_content}

Now create a significantly improved version."""

        try:
            r3_resp = await self.chat_fn(
                messages=[
                    {"role": "system", "content": REFINER_SYSTEM},
                    {"role": "user", "content": refiner_prompt},
                ],
                model=self.refiner_model,
                temperature=0.5,
                max_tokens=4096,
            )
            r3_content = r3_resp.content or ""
        except Exception as e:
            logger.error(f"Round 3 (refiner) failed: {e}")
            r3_content = r1_content  # fallback to original

        r3 = RoundResult(
            round_num=3, role="refiner",
            content=r3_content,
            duration_ms=int((time.monotonic() - r3_start) * 1000),
        )
        result.rounds.append(r3)

        if on_round:
            on_round(3, "Refiner", f"Done ({r3.duration_ms}ms, {len(r3_content)} chars)")

        result.total_duration_ms = int((time.monotonic() - start) * 1000)

        # Build improvement summary
        result.improvement_summary = self._build_summary(result)

        return result

    def _parse_scores(self, critic_text: str) -> dict[str, Any]:
        """Extract scores from critic output."""
        scores: dict[str, Any] = {}
        import re
        for line in critic_text.split("\n"):
            line_lower = line.lower().strip()
            for metric in ("accuracy", "completeness", "clarity", "practicality", "overall"):
                if metric in line_lower:
                    match = re.search(r'(\d+)\s*/\s*10', line)
                    if match:
                        scores[metric] = int(match.group(1))
        return scores

    def _build_summary(self, result: RefinementResult) -> str:
        """Build a human-readable improvement summary."""
        r1 = result.rounds[0] if len(result.rounds) > 0 else None
        r2 = result.rounds[1] if len(result.rounds) > 1 else None
        r3 = result.rounds[2] if len(result.rounds) > 2 else None

        parts = []
        if r1:
            parts.append(f"Round 1 (Initial): {len(r1.content)} chars, {r1.duration_ms}ms")
        if r2:
            scores = r2.quality_markers
            overall = scores.get("overall", "?")
            parts.append(f"Round 2 (Critic): score {overall}/10, {r2.duration_ms}ms")
        if r3:
            improvement = ""
            if r1 and r3:
                diff = len(r3.content) - len(r1.content)
                if diff > 0:
                    improvement = f" (+{diff} chars)"
                elif diff < 0:
                    improvement = f" ({diff} chars, more concise)"
            parts.append(f"Round 3 (Refined): {len(r3.content)} chars{improvement}, {r3.duration_ms}ms")

        parts.append(f"Total: {result.total_duration_ms}ms")
        return " | ".join(parts)
