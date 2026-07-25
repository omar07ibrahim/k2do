from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from k2do.labs import deepthink_trace


@pytest.fixture(scope="module")
def receipt() -> dict:
    return asyncio.run(deepthink_trace.run_trace_lab())


def test_trace_lab_verifies_production_orchestration(receipt: dict) -> None:
    assert receipt["status"] == "verified"
    assert receipt["router"]["status"] == "verified"
    assert [case["route"] for case in receipt["router"]["cases"]] == [
        "deepthink",
        "simple",
    ]

    scenarios = {scenario["id"]: scenario for scenario in receipt["scenarios"]}
    synthesis = scenarios["synthesis_with_thinker_fallback_and_timeout"]
    assert synthesis["fallback"]["path"] == [
        "primary_error",
        "fallback_success",
    ]
    assert synthesis["timeout"] == {
        "actor": "pragmatist",
        "provider_call_cancelled": True,
        "result": "categorical_timeout",
    }
    assert synthesis["judge"] == {
        "gate": "after_all_thinkers_terminal",
        "mode": "parsed_synthesis",
        "selected": "analyst",
    }
    assert synthesis["concurrency"]["provider_peak_calls"] == 3

    judge_fallback = scenarios["judge_failure_uses_longest_successful_thinker"]
    assert judge_fallback["judge"]["model_path"] == [
        "primary_error",
        "fallback_error",
    ]
    assert judge_fallback["judge"]["mode"] == "longest_successful_thinker"
    assert judge_fallback["judge"]["selected"] == "creative"

    cancellation = scenarios["caller_cancellation_drains_parallel_thinkers"]
    assert cancellation["cancellation"]["cancelled_provider_calls"] == 3
    assert cancellation["cancellation"]["judge_calls"] == 0
    assert cancellation["cleanup"] == {
        "active_provider_calls": 0,
        "barriers_incomplete": 0,
        "root_task": "cancelled",
    }


def test_trace_lab_dag_is_structured_and_connected(receipt: dict) -> None:
    for scenario in receipt["scenarios"]:
        node_ids = {node["id"] for node in scenario["call_dag"]["nodes"]}
        assert node_ids
        for edge in scenario["call_dag"]["edges"]:
            assert edge["from"] in node_ids
            assert edge["to"] in node_ids
            assert edge["relation"]


def test_trace_lab_receipt_is_deterministic(receipt: dict) -> None:
    first = deepthink_trace.render_receipt(receipt)
    second_receipt = asyncio.run(deepthink_trace.run_trace_lab())
    second = deepthink_trace.render_receipt(second_receipt)
    assert second == first


def test_each_scenario_refuses_a_non_deepthink_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepthink_trace, "classify_query", lambda _query: "simple")
    for scenario in (
        deepthink_trace._run_synthesis_scenario,
        deepthink_trace._run_judge_fallback_scenario,
        deepthink_trace._run_cancellation_scenario,
    ):
        with pytest.raises(
            deepthink_trace.TraceLabError,
            match="scenario_router_did_not_select_deepthink",
        ):
            asyncio.run(scenario())


def test_trace_lab_cli_is_stable_across_hash_seeds() -> None:
    repository = Path(__file__).resolve().parents[1]
    outputs: list[bytes] = []
    for seed in ("0", "1", "4294967295"):
        completed = subprocess.run(
            [sys.executable, "-m", "k2do.labs.deepthink_trace"],
            cwd=repository,
            env={
                "PYTHONHASHSEED": seed,
                "PYTHONIOENCODING": "utf-8",
            },
            check=False,
            capture_output=True,
            timeout=15,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        assert json.loads(completed.stdout)["status"] == "verified"
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_trace_lab_receipt_redacts_fixture_payloads(receipt: dict) -> None:
    rendered = deepthink_trace.render_receipt(receipt)
    lowered = rendered.lower()
    assert "synthetic-candidate" not in lowered
    assert "original question" not in lowered
    assert "your role" not in lowered
    assert "offline/primary" not in lowered
    assert "offline/fallback" not in lowered
    assert "/home/" not in lowered
    assert "/tmp/" not in lowered
    assert "/etc/" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "localhost" not in lowered
    assert "ghp_" not in lowered
    assert not re.search(r"\b\d+(?:\.\d+)?ms\b", lowered)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for key, value in _walk_items(receipt)
        if key.endswith("_sha256")
    )


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
def test_trace_lab_receipt_rejects_leak_canaries(receipt: dict, canary: str) -> None:
    tampered = deepcopy(receipt)
    tampered["evidence_boundary"]["provider"] = canary
    with pytest.raises(deepthink_trace.TraceLabError):
        deepthink_trace.render_receipt(tampered)


def test_trace_lab_receipt_rejects_unknown_structure(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["unexpected"] = "verified"
    with pytest.raises(
        deepthink_trace.TraceLabError,
        match="public_root_keys_changed",
    ):
        deepthink_trace.render_receipt(tampered)


def test_trace_lab_cli_emits_only_receipt(
    receipt: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_trace_lab() -> dict:
        return receipt

    monkeypatch.setattr(deepthink_trace, "run_trace_lab", fake_run_trace_lab)
    assert deepthink_trace.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == receipt


def test_trace_lab_cli_does_not_echo_rejected_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected = "private-prompt-or-credential"
    assert deepthink_trace.main(["--query", rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert rejected not in captured.err
    assert json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": "k2do.deepthink-trace-lab/v1",
        "status": "failed",
    }


def _walk_items(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)
