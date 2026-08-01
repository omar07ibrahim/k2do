from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from k2do.labs import agent_handoff_trace
from k2do.labs import handoff_evidence_recorder as recorder


@pytest.fixture(scope="module")
def receipt_bytes() -> bytes:
    receipt = asyncio.run(agent_handoff_trace.run_handoff_lab())
    return agent_handoff_trace.render_receipt(receipt).encode()


@pytest.fixture(scope="module")
def observation() -> dict[str, Any]:
    return {
        "communication_syscall_count": 0,
        "scope": recorder._NETWORK_OBSERVATION_SCOPE,
        "status": "no_communication_syscalls_observed",
        "syscalls": list(recorder.evidence_core._COMMUNICATION_SYSCALLS),
        "tool": "strace -- version 6.8",
    }


@pytest.fixture(scope="module")
def artifacts(receipt_bytes: bytes, observation: dict[str, Any]) -> dict[str, bytes]:
    return recorder._render_artifacts(receipt_bytes, observation)


def test_artifact_set_is_complete_and_deterministic(
    receipt_bytes: bytes,
    observation: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    assert set(artifacts) == set(recorder._ARTIFACT_MEDIA_TYPES)
    assert recorder._render_artifacts(receipt_bytes, observation) == artifacts
    assert artifacts["receipt.json"] == receipt_bytes
    assert artifacts["handoff-lab.txt"] == recorder._TRANSCRIPT_COMMAND.encode() + receipt_bytes


def test_visual_bytes_are_stable_across_hash_seeds() -> None:
    repository = Path(__file__).resolve().parents[1]
    program = """
import asyncio, json
from loguru import logger
from k2do.labs import agent_handoff_trace as lab
from k2do.labs import handoff_evidence_recorder as recorder
logger.disable('k2do.agent.loop')
receipt = asyncio.run(lab.run_handoff_lab())
receipt_bytes = lab.render_receipt(receipt).encode()
observation = {
    'communication_syscall_count': 0,
    'scope': recorder._NETWORK_OBSERVATION_SCOPE,
    'status': 'no_communication_syscalls_observed',
    'syscalls': list(recorder.evidence_core._COMMUNICATION_SYSCALLS),
    'tool': 'strace -- version 6.8',
}
artifacts = recorder._render_artifacts(receipt_bytes, observation)
print(json.dumps({name: recorder._sha256(data) for name, data in sorted(artifacts.items())}, sort_keys=True))
"""
    outputs: list[bytes] = []
    for seed in ("0", "1", "4294967295"):
        environment = dict(os.environ)
        environment.update({"PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"})
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            timeout=20,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def test_terminal_png_is_a_real_raster_of_the_capture(artifacts: dict[str, bytes]) -> None:
    with Image.open(io.BytesIO(artifacts["terminal.png"])) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width >= 1200
        assert image.height >= 1200
        assert image.info == {}


def test_workflow_gif_is_a_nine_frame_receipt_replay(artifacts: dict[str, bytes]) -> None:
    with Image.open(io.BytesIO(artifacts["workflow-demo.gif"])) as image:
        assert image.format == "GIF"
        assert image.size == (1200, 675)
        assert image.n_frames == 9
        assert image.info["loop"] == 0
        durations = []
        for frame in range(image.n_frames):
            image.seek(frame)
            durations.append(image.info["duration"])
        assert durations == [700] * 8 + [1600]


def test_svg_visuals_are_safe_and_receipt_derived(artifacts: dict[str, bytes]) -> None:
    for name in ("architecture.svg", "contract-matrix.svg", "terminal.svg", "tool-timeline.svg"):
        recorder.evidence_core._validate_svg(artifacts[name])
        lowered = artifacts[name].lower()
        assert b"<script" not in lowered
        assert b"<foreignobject" not in lowered
        assert b"https://" not in lowered
        assert lowered.count(b"http://") == 1
        assert b'xmlns="http://www.w3.org/2000/svg"' in lowered

    architecture = artifacts["architecture.svg"].decode()
    assert "message_bus" in architecture
    assert "parallel" in architecture
    assert "write_file" in architecture
    assert "session" in architecture


def test_captured_receipt_rejects_duplicate_keys(receipt_bytes: bytes) -> None:
    duplicate = receipt_bytes.replace(
        b'{\n  "cleanup":',
        b'{\n  "status": "verified",\n  "cleanup":',
        1,
    )
    with pytest.raises(recorder.EvidenceError, match="receipt_json_invalid"):
        recorder._validate_receipt(duplicate)


def test_raster_validation_rejects_tampering(artifacts: dict[str, bytes]) -> None:
    with pytest.raises(recorder.EvidenceError, match="raster_decode_failed"):
        recorder._validate_raster("terminal.png", artifacts["terminal.png"][:100])


def _fake_sources() -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    source_paths = ("pyproject.toml", "k2do/example.py")
    sources = {
        path: {
            "bytes": index + 1,
            "git_blob": f"{index + 1:040x}",
            "git_mode": "100644",
            "sha256": f"{index + 1:064x}",
        }
        for index, path in enumerate(source_paths)
    }
    return source_paths, sources


def test_manifest_is_canonical_and_strict(
    artifacts: dict[str, bytes],
    observation: dict[str, Any],
) -> None:
    source_paths, sources = _fake_sources()
    manifest = recorder._build_manifest(
        head="a" * 40,
        tree="b" * 40,
        sources=sources,
        artifacts=artifacts,
        network_observation=observation,
    )
    manifest_bytes = recorder._canonical_json(manifest)
    assert recorder._strict_manifest(manifest_bytes, source_paths) == manifest

    tampered = deepcopy(manifest)
    tampered["unexpected"] = True
    with pytest.raises(recorder.EvidenceError, match="manifest_structure_invalid"):
        recorder._strict_manifest(recorder._canonical_json(tampered), source_paths)


def test_manifest_rejects_artifact_hash_tampering(
    artifacts: dict[str, bytes],
    observation: dict[str, Any],
) -> None:
    source_paths, sources = _fake_sources()
    manifest = recorder._build_manifest(
        head="a" * 40,
        tree="b" * 40,
        sources=sources,
        artifacts=artifacts,
        network_observation=observation,
    )
    manifest["artifacts"]["terminal.png"]["sha256"] = "invalid"
    with pytest.raises(recorder.EvidenceError, match="manifest_artifact_entry_invalid"):
        recorder._strict_manifest(recorder._canonical_json(manifest), source_paths)


def test_linux_snapshot_capture_observes_no_communication_syscalls() -> None:
    if (
        recorder.platform.system() != "Linux"
        or not Path(recorder.evidence_core._STRACE_PATH).is_file()
    ):
        pytest.skip("trusted strace capture is Linux-only")
    repository = Path(__file__).resolve().parents[1]
    receipt, network = recorder._capture_receipt(repository)
    assert recorder._validate_receipt(receipt)["status"] == "verified"
    assert (
        recorder._sha256(receipt)
        == "0682e830944c812b111223f83d8730a8d4e2307c35fddb90b3e2503fbe0bde33"
    )
    assert network["communication_syscall_count"] == 0
    assert network["status"] == "no_communication_syscalls_observed"


def test_source_set_covers_pyproject_and_every_committed_package_blob() -> None:
    repository = Path(__file__).resolve().parents[1]
    head = (
        recorder.evidence_core._run_git(
            repository,
            ["rev-parse", "--verify", "HEAD"],
        )
        .decode()
        .strip()
    )
    source_paths = recorder._source_paths(repository, head)
    package_paths = [
        path
        for path, _mode, _object_id in recorder.evidence_core._committed_package_entries(
            repository,
            head,
        )
    ]
    assert source_paths == ("pyproject.toml", *package_paths)
    assert "k2do/labs/agent_handoff_trace.py" in source_paths
    assert "k2do/labs/trace_evidence_recorder.py" in source_paths
    assert all(path.startswith("k2do/") for path in source_paths[1:])


def test_atomic_publication_writes_exact_regular_files(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    files = {"one.txt": b"one", "two.json": b"{}\n"}
    observed: list[str] = []

    def inspect_stage(stage_name: str) -> None:
        observed.append(stage_name)
        stage = tmp_path / "docs" / stage_name
        assert {path.name for path in stage.iterdir()} == set(files)

    recorder._publish_artifacts(tmp_path, files, pre_publish=inspect_stage)
    destination = tmp_path / "docs" / recorder._OUTPUT_DIRECTORY
    assert observed and destination.is_dir()
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == files
    assert all(path.stat().st_mode & 0o111 == 0 for path in destination.iterdir())


def test_atomic_publication_refuses_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "docs" / recorder._OUTPUT_DIRECTORY
    destination.mkdir(parents=True)
    with pytest.raises(recorder.EvidenceError, match="evidence_directory_already_exists"):
        recorder._publish_artifacts(
            tmp_path,
            {"one.txt": b"one"},
            pre_publish=lambda _stage: None,
        )


def test_publication_guard_allows_only_the_private_stage(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "k2do").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "k2do" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "pyproject.toml", "k2do/fixture.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    head, tree = recorder.evidence_core._assert_clean_committed_head(tmp_path)
    source_paths = ("pyproject.toml", "k2do/fixture.py")
    sources = recorder._source_inventory(tmp_path, source_paths)
    stage_name = ".agent-handoff-evidence.stage-fixture"
    stage = tmp_path / "docs" / stage_name
    stage.mkdir()
    (stage / "one.txt").write_bytes(b"one")

    recorder._assert_publication_state(
        tmp_path,
        head=head,
        tree=tree,
        sources=sources,
        source_paths=source_paths,
        stage_name=stage_name,
        file_names=("one.txt",),
    )
    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(recorder.EvidenceError, match="worktree_changed_before_publication"):
        recorder._assert_publication_state(
            tmp_path,
            head=head,
            tree=tree,
            sources=sources,
            source_paths=source_paths,
            stage_name=stage_name,
            file_names=("one.txt",),
        )


def test_cli_rejects_arguments_without_echoing_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected = "private-path-or-token"
    assert recorder.main(["--output", rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert rejected not in captured.err
    assert json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": "k2do.handoff-evidence/v1",
        "status": "failed",
    }


def test_cli_projects_internal_failures_safely(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> dict[str, Any]:
        raise RuntimeError("private failure details")

    monkeypatch.setattr(recorder, "check", fail)
    assert recorder.main(["check"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private failure details" not in captured.err
    assert json.loads(captured.err) == {
        "failure": "internal_error",
        "schema": "k2do.handoff-evidence/v1",
        "status": "failed",
    }
