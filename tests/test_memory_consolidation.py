from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.providers.base import LLMProvider, LLMResponse
from k2do.session.manager import Session


class _ConsolidationProvider(LLMProvider):
    def __init__(self, payload: Any):
        super().__init__(api_key=None, api_base=None)
        self.payload = payload
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = tools, model, max_tokens, temperature
        self.seen_messages.append(messages)
        return LLMResponse(content=self.payload)

    def get_default_model(self) -> str:
        return "k2-think-v2/LLM360/K2-Think-V2"


@pytest.mark.asyncio
async def test_consolidation_handles_structured_session_content(tmp_path: Path) -> None:
    provider = _ConsolidationProvider(
        '{"history_entry":"Summary ok","memory_update":{"facts":["prefers wav"]}}'
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    session = Session(key="cli:demo")
    session.messages = [
        {
            "role": "user",
            "content": {"note": "value"},
            "timestamp": "2026-02-17T20:15:08",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "generated plan"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
            "timestamp": "2026-02-17T20:15:10",
            "tools_used": ["deepthink"],
        },
    ]

    await loop._consolidate_memory(session, archive_all=True)

    history = (tmp_path / "memory" / "HISTORY.md").read_text(encoding="utf-8")
    long_term = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")

    assert "Summary ok" in history
    assert '"facts"' in long_term
    assert "prefers wav" in long_term

    prompt = provider.seen_messages[0][1]["content"]
    assert isinstance(prompt, str)
    assert '"note": "value"' in prompt
    assert "generated plan" in prompt


@pytest.mark.asyncio
async def test_consolidation_accepts_dict_response_content(tmp_path: Path) -> None:
    provider = _ConsolidationProvider(
        {"history_entry": "Entry from dict payload", "memory_update": "updated memory"}
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="k2-think-v2/LLM360/K2-Think-V2",
        fallback_model="k2-v2-instruct/LLM360/K2-V2-Instruct",
        deepthink_enabled=False,
    )

    session = Session(key="cli:demo")
    session.messages = [
        {
            "role": "user",
            "content": "hello",
            "timestamp": "2026-02-17T20:16:00",
        }
    ]

    await loop._consolidate_memory(session, archive_all=True)

    history = (tmp_path / "memory" / "HISTORY.md").read_text(encoding="utf-8")
    long_term = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")

    assert "Entry from dict payload" in history
    assert long_term == "updated memory"
