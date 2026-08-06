from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from k2do.labs import mcp_fault_lab


@pytest.fixture(scope="module")
def receipt() -> dict:
    return asyncio.run(mcp_fault_lab.run_mcp_fault_lab())


def _scenarios(receipt: dict) -> dict[str, dict]:
    return {scenario["id"]: scenario for scenario in receipt["scenarios"]}


def _walk_items(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)


def test_lab_exercises_the_pinned_production_surface(receipt: dict) -> None:
    assert receipt["status"] == "verified"
    assert receipt["protocol"] == {
        "audit": "private_authenticated_unix",
        "client": "mcp-python-sdk-2.0.0",
        "mode": "auto",
        "negotiated": "2026-07-28",
        "transport": "stdio_ndjson",
    }
    assert receipt["production_surface"] == [
        "AgentLoop.mcp_lifespan",
        "connect_mcp_servers",
        "MCPToolWrapper.execute",
        "ToolRegistry",
    ]
    assert receipt["invariants"] == {
        "atomic_catalog_publication": True,
        "caller_cancellation_propagated": True,
        "categorical_remote_errors": True,
        "idempotent_close": True,
        "same_loop_restart": True,
    }


def test_happy_path_uses_modern_discovery_and_stable_restart(receipt: dict) -> None:
    happy = _scenarios(receipt)["happy_discovery_call_restart"]
    assert happy["generations"] == 2
    assert happy["wire_path"] == ["server/discover", "tools/list", "tools/call"]
    assert happy["close_calls"] == 4
    assert happy["recovery"] == "same_loop_restart"
    assert happy["cleanup"] == {
        "processes_reaped": 2,
        "registry_restored": True,
        "stdin_eof": 2,
    }


def test_catalog_failure_isolated_from_healthy_peer(receipt: dict) -> None:
    isolation = _scenarios(receipt)["catalog_failure_isolated"]
    assert isolation["outcomes"] == [
        {"phase": "discovery", "status": "failed", "tool_count": 0},
        {"phase": "ready", "status": "connected", "tool_count": 1},
    ]
    assert isolation["failure_projection"] == "catalog_not_published"
    assert isolation["recovery"] == "healthy_peer_available"


def test_protocol_error_is_categorical_and_connection_recovers(receipt: dict) -> None:
    recovery = _scenarios(receipt)["protocol_error_then_same_connection_recovery"]
    assert recovery["call_path"] == ["categorical_error", "same_connection_success"]
    assert recovery["raw_fault_visible"] is False
    assert recovery["recovery"] == "same_connection_success"


def test_repeated_call_cancellation_emits_courtesy_cancel_and_recovers(
    receipt: dict,
) -> None:
    cancellation = _scenarios(receipt)["repeated_call_cancellation_recovers"]
    assert cancellation["cancelled_calls"] == 2
    assert cancellation["courtesy_cancellations"] == 2
    assert cancellation["cancellation"] == "propagated"
    assert cancellation["recovery"] == "same_connection_success"


def test_startup_cancellation_reaps_before_same_loop_restart(receipt: dict) -> None:
    startup = _scenarios(receipt)["startup_cancellation_reaps_then_restarts"]
    assert startup["catalog_after_cancel"] == 0
    assert startup["cancellation"] == "propagated"
    assert startup["generations"] == 2
    assert startup["recovery"] == "same_loop_restart"


def test_scope_cancellation_drains_borrower_before_restart(receipt: dict) -> None:
    scope = _scenarios(receipt)["scope_cancellation_drains_borrower_then_restarts"]
    assert scope["borrowers_cancelled"] == 1
    assert scope["courtesy_cancellations"] == 1
    assert scope["cancellation"] == "owner_and_borrower_propagated"
    assert scope["recovery"] == "same_loop_restart"


def test_every_process_generation_is_reaped_and_registry_is_restored(receipt: dict) -> None:
    scenarios = receipt["scenarios"]
    assert sum(scenario["generations"] for scenario in scenarios) == 10
    assert sum(scenario["cleanup"]["processes_reaped"] for scenario in scenarios) == 10
    assert sum(scenario["cleanup"]["stdin_eof"] for scenario in scenarios) == 10
    assert all(scenario["cleanup"]["registry_restored"] for scenario in scenarios)


def test_receipt_contains_only_deterministic_public_projection(receipt: dict) -> None:
    rendered = mcp_fault_lab.render_receipt(receipt)
    lowered = rendered.casefold()
    assert "k2do-fixture-raw-fault-sentinel" not in lowered
    assert "audit.sock" not in lowered
    assert "requestid" not in lowered
    assert "request_id" not in lowered
    assert "process_id" not in lowered
    assert '"pid"' not in lowered
    assert "/home/" not in lowered
    assert "/tmp/" not in lowered
    assert "localhost" not in lowered
    assert "127.0.0.1" not in lowered
    assert "ghp_" not in lowered
    assert not re.search(r"\b\d+(?:\.\d+)?ms\b", lowered)
    assert not re.search(r'"(?:elapsed|latency|duration)[^\"]*"', lowered)
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for key, value in _walk_items(receipt)
        if key.endswith("_sha256")
    )


def test_receipt_is_stable_across_fresh_process_generations(receipt: dict) -> None:
    first = mcp_fault_lab.render_receipt(receipt)
    second = mcp_fault_lab.render_receipt(asyncio.run(mcp_fault_lab.run_mcp_fault_lab()))
    assert second == first


@pytest.mark.parametrize(
    "canary",
    [
        "ghp_CANARY0123456789",
        "/etc/passwd",
        "/tmp/private-audit.sock",
        r"C:\Users\Canary\secret.txt",
        r"\\server\share\private.txt",
        "localhost:8080",
        "https://private.invalid/path",
        "k2do-fixture-raw-fault-sentinel",
    ],
)
def test_receipt_rejects_private_or_location_canaries(receipt: dict, canary: str) -> None:
    tampered = deepcopy(receipt)
    tampered["evidence_boundary"]["public_projection"] = canary
    with pytest.raises(mcp_fault_lab.MCPFaultLabError):
        mcp_fault_lab.render_receipt(tampered)


def test_receipt_rejects_unknown_structure(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["unexpected"] = "verified"
    with pytest.raises(
        mcp_fault_lab.MCPFaultLabError,
        match="public_root_keys_changed",
    ):
        mcp_fault_lab.render_receipt(tampered)


def test_cli_is_stable_across_hash_seeds() -> None:
    repository = Path(__file__).resolve().parents[1]
    outputs: list[bytes] = []
    for seed in ("0", "4294967295"):
        completed = subprocess.run(
            [sys.executable, "-m", "k2do.labs.mcp_fault_lab"],
            cwd=repository,
            env={
                "PYTHONHASHSEED": seed,
                "PYTHONIOENCODING": "utf-8",
            },
            check=False,
            capture_output=True,
            timeout=60,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        assert json.loads(completed.stdout)["status"] == "verified"
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_cli_emits_only_receipt(
    receipt: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_mcp_fault_lab() -> dict:
        return receipt

    monkeypatch.setattr(mcp_fault_lab, "run_mcp_fault_lab", fake_run_mcp_fault_lab)
    assert mcp_fault_lab.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == receipt


def test_cli_does_not_echo_rejected_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    rejected = "private-token-or-path"
    assert mcp_fault_lab.main([rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert rejected not in captured.err
    assert json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": "k2do.mcp-fault-lab/v1",
        "status": "failed",
    }
