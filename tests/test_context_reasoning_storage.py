from pathlib import Path

from k2do.agent.context import ContextBuilder


def test_context_builder_skips_reasoning_by_default(tmp_path: Path) -> None:
    builder = ContextBuilder(tmp_path)
    messages: list[dict] = []

    builder.add_assistant_message(
        messages,
        content="ok",
        reasoning_content="internal reasoning",
    )

    assert messages[0]["content"] == "ok"
    assert "reasoning_content" not in messages[0]


def test_context_builder_can_store_reasoning_when_enabled(tmp_path: Path) -> None:
    builder = ContextBuilder(tmp_path, store_reasoning=True)
    messages: list[dict] = []

    builder.add_assistant_message(
        messages,
        content="ok",
        reasoning_content="internal reasoning",
    )

    assert messages[0]["reasoning_content"] == "internal reasoning"
