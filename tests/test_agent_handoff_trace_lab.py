from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from k2do.agent.loop import AgentLoop
from k2do.bus.queue import MessageBus
from k2do.labs import agent_handoff_trace
from k2do.session.manager import SessionManager


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return asyncio.run(agent_handoff_trace.run_handoff_lab())


def test_handoff_lab_verifies_the_full_production_path(receipt: dict[str, Any]) -> None:
    assert receipt["status"] == "verified"
    assert receipt["route"] == {
        "classification": "deepthink",
        "complexity_threshold": 0.6,
        "input_sha256": "2a4cbcdcdc8884d7e48750d70c407174632ea5225856f50430474011864fa58c",
    }
    assert receipt["deepthink"]["provider_peak_calls"] == 3
    assert receipt["deepthink"]["judge_gate"] == "after_all_thinkers_terminal"
    assert receipt["handoff"]["tool_sequence"] == ["write_file", "read_file"]
    assert receipt["session"]["tools_used"] == [
        "deepthink",
        "write_file",
        "read_file",
    ]
    assert receipt["message_bus"] == {
        "inbound_messages": 1,
        "outbound_messages": 1,
        "queues": "drained",
    }
    assert receipt["cleanup"] == {
        "active_provider_calls": 0,
        "temporary_workspace": "removed",
    }


def test_workflow_uses_real_filesystem_tools_and_disk_session(tmp_path: Path) -> None:
    observation = asyncio.run(agent_handoff_trace._execute_workflow(tmp_path))

    artifact = tmp_path / agent_handoff_trace._RELATIVE_PATH
    assert artifact.read_text(encoding="utf-8") == agent_handoff_trace._ARTIFACT_CONTENT
    assert observation.artifact_bytes == artifact.stat().st_size
    assert observation.tool_sequence == ["write_file", "read_file"]

    loaded = SessionManager(tmp_path).get_or_create(agent_handoff_trace._SESSION_KEY)
    assert [message["role"] for message in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[-1]["tools_used"] == [
        "deepthink",
        "write_file",
        "read_file",
    ]


def test_handoff_receipt_is_byte_deterministic(receipt: dict[str, Any]) -> None:
    first = agent_handoff_trace.render_receipt(receipt)
    second = agent_handoff_trace.render_receipt(asyncio.run(agent_handoff_trace.run_handoff_lab()))
    assert second == first


def test_handoff_cli_is_stable_across_hash_seeds() -> None:
    repository = Path(__file__).resolve().parents[1]
    outputs: list[bytes] = []
    for seed in ("0", "1", "4294967295"):
        completed = subprocess.run(
            [sys.executable, "-m", "k2do.labs.agent_handoff_trace"],
            cwd=repository,
            env={"PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"},
            check=False,
            capture_output=True,
            timeout=15,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        assert json.loads(completed.stdout)["status"] == "verified"
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_handoff_receipt_contains_only_public_projections(receipt: dict[str, Any]) -> None:
    rendered = agent_handoff_trace.render_receipt(receipt)
    lowered = rendered.lower()
    assert "design, implement" not in lowered
    assert "handoff-proof:v1" not in lowered
    assert "workspace handoff verified" not in lowered
    assert "offline/handoff" not in lowered
    assert "original question" not in lowered
    assert "your role" not in lowered
    assert "/home/" not in lowered
    assert "/tmp/" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "localhost" not in lowered
    assert "ghp_" not in lowered
    assert not re.search(r"\b\d+(?:\.\d+)?ms\b", lowered)


@pytest.mark.parametrize(
    "canary",
    [
        "ghp_CANARY0123456789",
        "/etc/passwd",
        r"C:\Users\Canary\secret.txt",
        r"\\server\share\private.txt",
        "localhost:8080",
        "ws://localhost/socket",
        "grpc://127.0.0.1:50051",
    ],
)
def test_handoff_receipt_rejects_leak_canaries(
    receipt: dict[str, Any],
    canary: str,
) -> None:
    tampered = deepcopy(receipt)
    tampered["evidence_boundary"]["provider"] = canary
    with pytest.raises(agent_handoff_trace.HandoffTraceError):
        agent_handoff_trace.render_receipt(tampered)


def test_handoff_receipt_rejects_unknown_structure(receipt: dict[str, Any]) -> None:
    tampered = deepcopy(receipt)
    tampered["unexpected"] = "verified"
    with pytest.raises(
        agent_handoff_trace.HandoffTraceError,
        match="public_root_keys_changed",
    ):
        agent_handoff_trace.render_receipt(tampered)


def test_handoff_receipt_rejects_an_invalid_digest(receipt: dict[str, Any]) -> None:
    tampered = deepcopy(receipt)
    tampered["handoff"]["artifact"]["sha256"] = "not-a-digest"
    with pytest.raises(
        agent_handoff_trace.HandoffTraceError,
        match="public_digest_invalid",
    ):
        agent_handoff_trace.render_receipt(tampered)


def test_provider_rejects_a_changed_tool_catalog(tmp_path: Path) -> None:
    provider = agent_handoff_trace._StrictHandoffProvider(tmp_path)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model=agent_handoff_trace._MODEL,
        fallback_model=agent_handoff_trace._MODEL,
        brave_api_key="offline-fixture-not-a-credential",
        restrict_to_workspace=True,
    )
    catalog = loop.tools.get_definitions()
    provider._validate_tool_catalog(catalog)
    tampered = deepcopy(catalog)
    tampered[0], tampered[1] = tampered[1], tampered[0]
    with pytest.raises(
        agent_handoff_trace.HandoffTraceError,
        match="tool_catalog_names_changed",
    ):
        provider._validate_tool_catalog(tampered)


def test_provider_rejects_an_unallowlisted_model(tmp_path: Path) -> None:
    provider = agent_handoff_trace._StrictHandoffProvider(tmp_path)
    with pytest.raises(
        agent_handoff_trace.HandoffTraceError,
        match="provider_model_changed",
    ):
        asyncio.run(provider.chat([], model="remote/unexpected"))


def test_handoff_cli_emits_only_the_receipt(
    receipt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_handoff_lab() -> dict[str, Any]:
        return receipt

    monkeypatch.setattr(agent_handoff_trace, "run_handoff_lab", fake_run_handoff_lab)
    assert agent_handoff_trace.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == receipt


def test_handoff_cli_does_not_echo_rejected_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected = "private-prompt-or-credential"
    assert agent_handoff_trace.main(["--query", rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert rejected not in captured.err
    assert json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": "k2do.agent-handoff-trace/v1",
        "status": "failed",
    }


def test_handoff_cli_projects_internal_failures_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail() -> dict[str, Any]:
        raise RuntimeError("private failure details")

    monkeypatch.setattr(agent_handoff_trace, "run_handoff_lab", fail)
    assert agent_handoff_trace.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private failure details" not in captured.err
    assert json.loads(captured.err) == {
        "failure": "verification_failed",
        "schema": "k2do.agent-handoff-trace/v1",
        "status": "failed",
    }


def test_temporary_workspace_is_removed_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_temporary_directory = agent_handoff_trace.tempfile.TemporaryDirectory
    workspaces: list[Path] = []

    def tracked_temporary_directory(*args: Any, **kwargs: Any):
        temporary = real_temporary_directory(*args, **kwargs)
        workspaces.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(
        agent_handoff_trace.tempfile,
        "TemporaryDirectory",
        tracked_temporary_directory,
    )
    asyncio.run(agent_handoff_trace.run_handoff_lab())
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_temporary_workspace_is_removed_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspaces: list[Path] = []

    async def fail(workspace: Path):
        workspaces.append(workspace)
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(agent_handoff_trace, "_execute_workflow", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        asyncio.run(agent_handoff_trace.run_handoff_lab())
    assert len(workspaces) == 1
    assert not workspaces[0].exists()
