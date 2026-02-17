import asyncio

import pytest

from k2do.agent.deepthink import DeepThinkEngine
from k2do.providers.base import LLMResponse


@pytest.mark.asyncio
async def test_deepthink_thinker_timeout_returns_error_result() -> None:
    async def slow_chat(**kwargs) -> LLMResponse:
        _ = kwargs
        await asyncio.sleep(2.0)
        return LLMResponse(content="late")

    engine = DeepThinkEngine(
        chat_fn=slow_chat,
        model="k2-think-v2/LLM360/K2-Think-V2",
        thinkers=[{"name": "Analyst", "temperature": 0.3, "role_prompt": "Analyze"}],
        max_agents=1,
        thinker_timeout_s=0.5,
        judge_timeout_s=0.5,
    )

    result = await engine.think("hard task")

    assert len(result.thinker_results) == 1
    assert result.thinker_results[0].error is not None
    assert "Timed out" in result.thinker_results[0].error
