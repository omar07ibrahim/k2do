from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from k2do.cli import commands
from k2do.config import loader


def test_interactive_route_recognizes_explicit_mcp_without_prefix_collisions() -> None:
    assert commands._interactive_route("/mcp retrieve release notes") == "mcp"
    assert commands._interactive_route("/McP\tstatus") == "mcp"
    assert commands._interactive_route("/mcp") == "mcp"
    assert commands._interactive_route("/mcproxy") != "mcp"
    assert commands._interactive_route("/deepthinker") != "deepthink"
    assert commands._interactive_route("/refinement") != "refine"


def test_channel_status_never_renders_a_token_prefix(monkeypatch) -> None:
    secret = "1234567890:PRIVATE-TELEGRAM-TOKEN"
    config = SimpleNamespace(
        channels=SimpleNamespace(
            telegram=SimpleNamespace(enabled=True, token=secret),
        ),
    )
    recording_console = Console(record=True, width=100)
    monkeypatch.setattr(loader, "load_config", lambda: config)
    monkeypatch.setattr(commands, "console", recording_console)

    commands.channels_status()

    rendered = recording_console.export_text()
    assert "configured" in rendered
    assert secret not in rendered
    assert secret[:10] not in rendered
