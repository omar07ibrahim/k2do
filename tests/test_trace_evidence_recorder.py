from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest

from k2do.labs import deepthink_trace
from k2do.labs import trace_evidence_recorder as recorder

_RECEIPT_SHA256 = "f66e1db30dd79d77318a20ae86a0aa450a663e8f5cdff351557ca6fd1d0da80d"


@pytest.fixture(scope="module")
def receipt_bytes() -> bytes:
    receipt = asyncio.run(deepthink_trace.run_trace_lab())
    return deepthink_trace.render_receipt(receipt).encode("utf-8")


@pytest.fixture(scope="module")
def observation() -> dict:
    return {
        "communication_syscall_count": 0,
        "scope": recorder._NETWORK_OBSERVATION_SCOPE,
        "status": "no_communication_syscalls_observed",
        "syscalls": list(recorder._COMMUNICATION_SYSCALLS),
        "tool": "strace -- version 6.8",
    }


def _temporary_directory() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix=".trace-evidence-test-",
        dir=Path.cwd(),
    ) as directory:
        yield Path(directory)


def _git_environment() -> dict[str, str]:
    return {
        "GIT_AUTHOR_EMAIL": "31526072+omar07ibrahim@users.noreply.github.com",
        "GIT_AUTHOR_NAME": "Omar Ibrahim",
        "GIT_COMMITTER_EMAIL": "31526072+omar07ibrahim@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "Omar Ibrahim",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }


def _run_git(repository: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=_git_environment(),
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def _commit_fixture(repository: Path, message: str) -> None:
    script = r"""find . -type d \( -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ \) -prune -exec rm -r {} +
GIT_AUTHOR_NAME='Omar Ibrahim' GIT_AUTHOR_EMAIL='31526072+omar07ibrahim@users.noreply.github.com' GIT_COMMITTER_NAME='Omar Ibrahim' GIT_COMMITTER_EMAIL='31526072+omar07ibrahim@users.noreply.github.com' git commit -q -m "$1"
"""
    completed = subprocess.run(
        ["/bin/bash", "-c", script, "fixture-commit", message],
        cwd=repository,
        env=_git_environment(),
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )


def _write_source_fixture(repository: Path) -> None:
    for index, relative_path in enumerate(recorder._SOURCE_PATHS):
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            f"# fixture source {index}: {relative_path}\n",
            encoding="utf-8",
        )
    (repository / "docs").mkdir()


def _initialize_repository(repository: Path) -> None:
    _run_git(repository, "init", "-q")
    _run_git(repository, "add", ".")
    _commit_fixture(repository, "Create recorder fixture")


def _fake_capture(
    receipt_bytes: bytes,
    observation: dict,
):
    def capture(_repository: Path) -> tuple[bytes, dict]:
        return receipt_bytes, deepcopy(observation)

    return capture


def test_rendered_artifacts_are_receipt_derived_and_safe(
    receipt_bytes: bytes,
    observation: dict,
) -> None:
    artifacts = recorder._render_artifacts(receipt_bytes, observation)
    assert set(artifacts) == set(recorder._ARTIFACT_MEDIA_TYPES)
    assert recorder._sha256(receipt_bytes) == _RECEIPT_SHA256
    assert artifacts["trace-lab.txt"] == (
        recorder._TRANSCRIPT_COMMAND.encode("utf-8") + receipt_bytes
    )

    for name in (
        "trace-lab.svg",
        "call-dag.svg",
        "orchestration-properties.svg",
    ):
        root = ET.fromstring(artifacts[name])
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib["role"] == "img"
        assert b"<script" not in artifacts[name]
        assert b"href=" not in artifacts[name]

    terminal = artifacts["trace-lab.svg"].decode()
    assert "status                 verified" in terminal
    assert "router complex/simple  deepthink / simple" in terminal
    assert "network observation   0 observed in listed strace set" in terminal
    assert _RECEIPT_SHA256 in terminal

    dag = artifacts["call-dag.svg"].decode()
    for value in (
        "analyst · primary",
        "analyst · fallback",
        "pragmatist · primary",
        "fallback_on_error",
        "judge_gate",
        "timeout",
        "synthesis",
    ):
        assert value in dag

    properties = artifacts["orchestration-properties.svg"].decode()
    for value in (
        "Synthesis path",
        "Judge degradation",
        "Caller cancellation",
        "timeout           pragmatist / categorical_timeout",
        "provider cancelled true",
        "cancelled calls   3",
        "judge calls       0",
        "active after      0",
    ):
        assert value in properties

    combined = b"\n".join(artifacts.values()).lower()
    for forbidden in (
        b"offline/primary",
        b"offline/fallback",
        b"synthetic-candidate",
        b"synthetic-control-credential",
        b"/home/",
        b"/tmp/",
        b"github_pat_",
        b"ghp_",
    ):
        assert forbidden not in combined
    transcript_surface = (artifacts["receipt.json"] + b"\n" + artifacts["trace-lab.txt"]).lower()
    assert b"http://" not in transcript_surface
    assert b"https://" not in transcript_surface


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("strace") is None,
    reason="the production communication-syscall probe requires Linux strace",
)
def test_real_capture_observes_no_communication_syscalls(
    receipt_bytes: bytes,
) -> None:
    captured, observation = recorder._capture_receipt(Path.cwd())
    assert captured == receipt_bytes
    assert recorder._sha256(captured) == _RECEIPT_SHA256
    assert observation["communication_syscall_count"] == 0
    assert observation["status"] == "no_communication_syscalls_observed"
    assert observation["syscalls"] == list(recorder._COMMUNICATION_SYSCALLS)


def test_record_and_check_are_manifest_bound_and_squash_safe(
    receipt_bytes: bytes,
    observation: dict,
) -> None:
    capture = _fake_capture(receipt_bytes, observation)
    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)

        result = recorder.record(repository, capture=capture)
        assert result == {
            "artifact_count": 5,
            "receipt_sha256": _RECEIPT_SHA256,
            "schema": recorder._SCHEMA,
            "source_commit": result["source_commit"],
            "status": "recorded",
        }
        output = repository / "docs" / recorder._OUTPUT_DIRECTORY
        assert {path.name for path in output.iterdir()} == {
            *recorder._ARTIFACT_MEDIA_TYPES,
            "manifest.json",
        }

        checked = recorder.check(
            repository,
            capture=capture,
            recapture=True,
        )
        assert checked["status"] == "verified"
        assert checked["capture_object"] == "verified"
        assert checked["fresh_capture"] == "verified"
        assert checked["receipt_sha256"] == _RECEIPT_SHA256
        assert checked["source_blob_count"] == len(recorder._SOURCE_PATHS)

        # A new root commit with identical source blobs and evidence must still
        # verify even though the manifest's original source commit is absent.
        for squashed in _temporary_directory():
            for relative_path in recorder._SOURCE_PATHS:
                destination = squashed / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repository / relative_path, destination)
            shutil.copytree(
                output,
                squashed / "docs" / recorder._OUTPUT_DIRECTORY,
            )
            _initialize_repository(squashed)
            squashed_result = recorder.check(
                squashed,
                capture=capture,
                recapture=True,
            )
            assert squashed_result["status"] == "verified"
            assert squashed_result["capture_object"] == "unavailable_after_squash"
            assert squashed_result["receipt_sha256"] == _RECEIPT_SHA256


def test_check_rejects_extra_tampered_and_symlinked_artifacts(
    receipt_bytes: bytes,
    observation: dict,
) -> None:
    capture = _fake_capture(receipt_bytes, observation)
    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)
        recorder.record(repository, capture=capture)
        output = repository / "docs" / recorder._OUTPUT_DIRECTORY

        extra = output / "extra.txt"
        extra.write_text("unexpected\n", encoding="utf-8")
        with pytest.raises(recorder.EvidenceError, match="evidence_file_set_changed"):
            recorder.check(repository, capture=capture, recapture=True)
        extra.unlink()

        receipt_path = output / "receipt.json"
        original_receipt = receipt_path.read_bytes()
        receipt_path.write_bytes(original_receipt + b" ")
        with pytest.raises(recorder.EvidenceError, match="artifact_identity_mismatch"):
            recorder.check(repository, capture=capture, recapture=True)
        receipt_path.write_bytes(original_receipt)

        visual_path = output / "trace-lab.svg"
        original_visual = visual_path.read_bytes()
        visual_path.unlink()
        visual_path.symlink_to("receipt.json")
        with pytest.raises(recorder.EvidenceError, match="evidence_file_not_safe"):
            recorder.check(repository, capture=capture, recapture=True)
        visual_path.unlink()
        visual_path.write_bytes(original_visual)
        visual_path.chmod(0o644)

        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["unexpected"] = True
        manifest_path.write_bytes(recorder._canonical_json(manifest))
        with pytest.raises(recorder.EvidenceError, match="manifest_structure_invalid"):
            recorder.check(repository, capture=capture, recapture=True)


def test_record_rejects_dirty_and_mid_capture_source_changes(
    receipt_bytes: bytes,
    observation: dict,
) -> None:
    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)
        source = repository / recorder._SOURCE_PATHS[0]
        source.write_text("dirty source\n", encoding="utf-8")
        with pytest.raises(recorder.EvidenceError, match="record_requires_clean_tree"):
            recorder.record(
                repository,
                capture=_fake_capture(receipt_bytes, observation),
            )
        assert not (repository / "docs" / recorder._OUTPUT_DIRECTORY).exists()

    for dirt_kind in ("untracked", "staged"):
        for repository in _temporary_directory():
            _write_source_fixture(repository)
            _initialize_repository(repository)
            if dirt_kind == "untracked":
                (repository / "untracked-canary").write_text(
                    "private\n",
                    encoding="utf-8",
                )
            else:
                source = repository / recorder._SOURCE_PATHS[0]
                source.write_text("staged source\n", encoding="utf-8")
                _run_git(repository, "add", recorder._SOURCE_PATHS[0])
            with pytest.raises(
                recorder.EvidenceError,
                match="record_requires_clean_tree",
            ):
                recorder.record(
                    repository,
                    capture=_fake_capture(receipt_bytes, observation),
                )
            assert not (repository / "docs" / recorder._OUTPUT_DIRECTORY).exists()

    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)

        def mutating_capture(root: Path) -> tuple[bytes, dict]:
            (root / recorder._SOURCE_PATHS[0]).write_text(
                "changed during capture\n",
                encoding="utf-8",
            )
            return receipt_bytes, deepcopy(observation)

        with pytest.raises(
            recorder.EvidenceError,
            match="source_worktree_content_mismatch",
        ):
            recorder.record(repository, capture=mutating_capture)
        assert not (repository / "docs" / recorder._OUTPUT_DIRECTORY).exists()


def test_public_cli_rejects_arguments_without_echoing_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "private-prompt-or-token"
    assert recorder.main(["record", canary]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert canary not in captured.err
    assert json.loads(captured.err) == {
        "failure": "invalid_invocation",
        "schema": recorder._SCHEMA,
        "status": "failed",
    }


def test_renderer_refuses_an_unknown_synthesis_dag(
    receipt_bytes: bytes,
) -> None:
    receipt = json.loads(receipt_bytes)
    tampered = deepcopy(receipt)
    scenario = next(
        item
        for item in tampered["scenarios"]
        if item["id"] == "synthesis_with_thinker_fallback_and_timeout"
    )
    scenario["call_dag"]["nodes"][0]["id"] = "unexpected"
    with pytest.raises(recorder.EvidenceError, match="synthesis_dag_shape_changed"):
        recorder._render_call_dag_svg(tampered)


def test_renderers_do_not_depend_on_host_line_separator(
    receipt_bytes: bytes,
    observation: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = recorder._render_artifacts(receipt_bytes, observation)
    monkeypatch.setattr(recorder.os, "linesep", "\r\n")
    assert recorder._render_artifacts(receipt_bytes, observation) == expected


def test_duplicate_manifest_keys_are_rejected() -> None:
    duplicate = b'{"schema":"first","schema":"second"}\n'
    with pytest.raises(recorder.EvidenceError, match="manifest_json_invalid"):
        recorder._strict_manifest(duplicate)


def test_preexisting_collision_leaves_are_never_deleted(
    receipt_bytes: bytes,
    observation: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "0123456789abcdef"
    monkeypatch.setattr(recorder.secrets, "token_hex", lambda _size: token)
    capture = _fake_capture(receipt_bytes, observation)

    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)
        snapshot_name = f".trace-evidence-snapshot-{os.getpid()}-{token}"
        collision = repository / snapshot_name
        collision.mkdir()
        sentinel = collision / "sentinel.txt"
        sentinel.write_text("owned by another actor\n", encoding="utf-8")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with pytest.raises(recorder.EvidenceError, match="snapshot_root_create_failed"):
            with recorder._committed_snapshot(repository, head):
                pytest.fail("a colliding snapshot leaf was entered")
        assert sentinel.read_text(encoding="utf-8") == "owned by another actor\n"

    for repository in _temporary_directory():
        (repository / "docs").mkdir()
        stage_name = f".{recorder._OUTPUT_DIRECTORY}.stage-{os.getpid()}-{token}"
        collision = repository / "docs" / stage_name
        collision.mkdir()
        sentinel = collision / "sentinel.txt"
        sentinel.write_text("must survive\n", encoding="utf-8")
        bundle = {name: b"fixture\n" for name in {*recorder._ARTIFACT_MEDIA_TYPES, "manifest.json"}}
        with pytest.raises(
            recorder.EvidenceError,
            match="artifact_publication_io_failed",
        ):
            recorder._publish_artifacts(repository, bundle)
        assert sentinel.read_text(encoding="utf-8") == "must survive\n"

    # The normal path remains intact after collision testing.
    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)
        assert recorder.record(repository, capture=capture)["status"] == "recorded"


@pytest.mark.parametrize("leaf_kind", ["file", "directory", "symlink", "fifo"])
def test_publication_refuses_every_preexisting_final_leaf(leaf_kind: str) -> None:
    for repository in _temporary_directory():
        docs = repository / "docs"
        docs.mkdir()
        target = docs / recorder._OUTPUT_DIRECTORY
        if leaf_kind == "file":
            target.write_text("existing\n", encoding="utf-8")
        elif leaf_kind == "directory":
            target.mkdir()
            (target / "sentinel").write_text("existing\n", encoding="utf-8")
        elif leaf_kind == "symlink":
            target.symlink_to("missing-target")
        else:
            os.mkfifo(target)
        bundle = {name: b"fixture\n" for name in {*recorder._ARTIFACT_MEDIA_TYPES, "manifest.json"}}
        with pytest.raises(
            recorder.EvidenceError,
            match="evidence_directory_already_exists",
        ):
            recorder._publish_artifacts(repository, bundle)
        assert target.exists() or target.is_symlink()


def test_bounded_runner_caps_output_and_kills_term_ignoring_descendants() -> None:
    environment = recorder._capture_environment()
    with pytest.raises(recorder.EvidenceError, match="capture_output_exceeded_limit"):
        recorder._run_bounded(
            [
                sys.executable,
                "-I",
                "-c",
                "import os; os.write(1, b'x' * 4096)",
            ],
            cwd=Path.cwd(),
            environment=environment,
            timeout_s=5,
            stdout_limit=128,
            stderr_limit=128,
        )
    with pytest.raises(recorder.EvidenceError, match="capture_timed_out"):
        recorder._run_bounded(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            cwd=Path.cwd(),
            environment=environment,
            timeout_s=0.1,
            stdout_limit=128,
            stderr_limit=128,
        )

    child_program = (
        "import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(30)"
    )
    leader_program = (
        "import subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-I','-c',{child_program!r}]);"
        "print(child.pid,flush=True);"
        "time.sleep(30)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-I", "-c", leader_program],
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        text=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    time.sleep(0.1)
    recorder._terminate_process_group(leader)
    leader.wait(timeout=2)
    leader.stdout.close()

    deadline = time.monotonic() + 2
    child_state = ""
    while time.monotonic() < deadline:
        try:
            child_state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            child_state = "absent"
            break
        if child_state == "Z":
            break
        time.sleep(0.02)
    assert child_state in {"absent", "Z"}


def test_competitor_wins_noreplace_without_partial_publication() -> None:
    for repository in _temporary_directory():
        docs = repository / "docs"
        docs.mkdir()
        bundle = {
            name: f"{name}\n".encode()
            for name in {*recorder._ARTIFACT_MEDIA_TYPES, "manifest.json"}
        }

        def competitor(stage_name: str) -> None:
            final = docs / recorder._OUTPUT_DIRECTORY
            final.mkdir()
            (final / "sentinel").write_text(
                "competitor publication\n",
                encoding="utf-8",
            )
            assert (docs / stage_name).is_dir()

        with pytest.raises(
            recorder.EvidenceError,
            match="evidence_directory_already_exists",
        ):
            recorder._publish_artifacts(
                repository,
                bundle,
                pre_publish=competitor,
            )
        final = docs / recorder._OUTPUT_DIRECTORY
        assert {path.name for path in final.iterdir()} == {"sentinel"}
        assert not list(docs.glob(f".{recorder._OUTPUT_DIRECTORY}.stage-*"))


def test_prepublication_extra_stage_file_fails_closed(
    receipt_bytes: bytes,
    observation: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _fake_capture(receipt_bytes, observation)
    original = recorder._assert_publication_state

    for repository in _temporary_directory():
        _write_source_fixture(repository)
        _initialize_repository(repository)

        def inject_extra(
            root: Path,
            *,
            expected_head: str,
            expected_tree: str,
            expected_sources,
            stage_name: str,
        ) -> None:
            stage = root / "docs" / stage_name
            (stage / "competitor-extra").write_text(
                "must not be silently deleted\n",
                encoding="utf-8",
            )
            original(
                root,
                expected_head=expected_head,
                expected_tree=expected_tree,
                expected_sources=expected_sources,
                stage_name=stage_name,
            )

        monkeypatch.setattr(recorder, "_assert_publication_state", inject_extra)
        with pytest.raises(
            recorder.EvidenceError,
            match="worktree_changed_before_publication",
        ):
            recorder.record(repository, capture=capture)
        assert not (repository / "docs" / recorder._OUTPUT_DIRECTORY).exists()
        stage_leaves = list((repository / "docs").glob(f".{recorder._OUTPUT_DIRECTORY}.stage-*"))
        assert len(stage_leaves) == 1
        assert {path.name for path in stage_leaves[0].iterdir()} == {"competitor-extra"}


def test_evidence_commit_survives_fresh_umask_restricted_clone(
    receipt_bytes: bytes,
    observation: dict,
) -> None:
    capture = _fake_capture(receipt_bytes, observation)
    for root in _temporary_directory():
        source = root / "source"
        source.mkdir()
        _write_source_fixture(source)
        _initialize_repository(source)
        recorder.record(source, capture=capture)
        _run_git(source, "add", "docs")
        _commit_fixture(source, "Publish fixture evidence")

        clone = root / "clone"
        previous_umask = os.umask(0o077)
        try:
            completed = subprocess.run(
                ["git", "clone", "-q", "--no-local", str(source), str(clone)],
                cwd=root,
                env=_git_environment(),
                check=False,
                capture_output=True,
                timeout=20,
            )
        finally:
            os.umask(previous_umask)
        assert completed.returncode == 0
        result = recorder.check(clone, capture=capture, recapture=True)
        assert result["status"] == "verified"
        assert result["capture_object"] == "verified"
        modes = {
            stat_result.st_mode & 0o777
            for stat_result in (
                path.stat() for path in (clone / "docs" / recorder._OUTPUT_DIRECTORY).iterdir()
            )
        }
        assert modes == {0o600}
