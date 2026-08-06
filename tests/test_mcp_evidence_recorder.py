from __future__ import annotations

import xml.etree.ElementTree as ET

from PIL import Image

from k2do.labs import mcp_evidence_recorder as recorder


def _receipt() -> dict:
    ids = recorder._SCENARIO_IDS
    recoveries = (
        "same_loop_restart",
        "healthy_peer_available",
        "same_connection_success",
        "same_connection_success",
        "same_loop_restart",
        "same_loop_restart",
    )
    generations = (2, 2, 1, 1, 2, 2)
    scenarios = []
    for scenario_id, recovery, generation_count in zip(
        ids,
        recoveries,
        generations,
        strict=True,
    ):
        scenario = {
            "id": scenario_id,
            "generations": generation_count,
            "recovery": recovery,
            "cleanup": {
                "processes_reaped": generation_count,
                "registry_restored": True,
                "stdin_eof": generation_count,
            },
        }
        if scenario_id == "repeated_call_cancellation_recovers":
            scenario["courtesy_cancellations"] = 2
        if scenario_id == "scope_cancellation_drains_borrower_then_restarts":
            scenario["cancellation"] = "owner_and_borrower_propagated"
        scenarios.append(scenario)
    return {
        "schema": "test-only-render-receipt",
        "status": "verified",
        "protocol": {
            "client": "mcp-python-sdk-2.0.0",
            "mode": "auto",
            "negotiated": "2026-07-28",
            "transport": "stdio_ndjson",
            "audit": "private_authenticated_unix",
        },
        "scenarios": scenarios,
        "invariants": {
            "atomic_catalog_publication": True,
            "categorical_remote_errors": True,
            "caller_cancellation_propagated": True,
            "idempotent_close": True,
            "same_loop_restart": True,
        },
        "evidence_boundary": {
            "fixture_io": ["stdio", "private_unix_audit"],
            "external_network": "no_external_network",
            "public_projection": "labels_counts_booleans_sha256",
            "raw_rpc_fields": 0,
            "wall_clock_metrics": 0,
        },
    }


def test_visual_renderer_is_deterministic_and_structurally_valid() -> None:
    receipt = _receipt()
    receipt_bytes = recorder._canonical_json(receipt)

    first = recorder._render_artifacts(receipt_bytes, receipt)
    second = recorder._render_artifacts(receipt_bytes, receipt)

    assert first == second
    assert set(first) == set(recorder._MEDIA_TYPES)
    for name in ("architecture.svg", "cancellation-timeline.svg", "fault-matrix.svg", "terminal.svg"):
        assert ET.fromstring(first[name]).tag.endswith("svg")
    with Image.open(recorder.io.BytesIO(first["terminal.png"])) as image:
        assert image.width == 1600
        assert image.height >= 500
    with Image.open(recorder.io.BytesIO(first["workflow-demo.gif"])) as image:
        assert image.n_frames == len(recorder._SCENARIO_IDS)


def test_publication_replaces_only_the_evidence_directory(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()

    recorder._publish(tmp_path, {"receipt.json": b"first\n"}, replace=False)
    assert (docs / recorder._OUTPUT_DIRECTORY / "receipt.json").read_bytes() == b"first\n"

    recorder._publish(tmp_path, {"receipt.json": b"second\n"}, replace=True)
    assert (docs / recorder._OUTPUT_DIRECTORY / "receipt.json").read_bytes() == b"second\n"
    assert not list(docs.glob(f".{recorder._OUTPUT_DIRECTORY}-*"))


def test_cli_rejects_private_arguments_without_reflection(capsys) -> None:
    rejected = "private-token-or-path"
    assert recorder.main([rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert rejected not in captured.err
    assert recorder.json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": "k2do.mcp-fault-evidence/v1",
        "status": "failed",
    }
