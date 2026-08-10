"""Publish and verify commit-bound evidence for the offline DeepThink trace lab.

The public ``record`` command is deliberately one-shot: it requires a clean
committed tree and an absent destination directory, captures the trace lab
under a bounded Linux ``strace`` communication-syscall probe, renders every
visual from that captured receipt, and atomically publishes the complete
directory without replacement.

Readers normally use ``check``. It validates descriptor-relative regular
files, artifact hashes, current source blob identities, receipt semantics, and
renderer output. On Linux hosts with ``strace``, it also executes a fresh
credential-free capture and compares the canonical receipt.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterator

from k2do.labs import deepthink_trace

_SCHEMA: Final = "k2do.trace-evidence/v1"
_MANIFEST_SCHEMA: Final = "k2do.trace-evidence-manifest/v1"
_OUTPUT_DIRECTORY: Final = "deepthink-trace-evidence"
_MAX_COMMAND_BYTES: Final = 256 * 1024
_MAX_ARTIFACT_BYTES: Final = 512 * 1024
_COMMAND_TIMEOUT_S: Final = 20.0
_RENAME_NOREPLACE: Final = 1

_SOURCE_PATHS: Final = (
    "pyproject.toml",
    "requirements-ci-py312.lock",
    "requirements-ci-py312.provenance.json",
    "k2do/__init__.py",
    "k2do/agent/__init__.py",
    "k2do/agent/deepthink.py",
    "k2do/agent/router.py",
    "k2do/labs/__init__.py",
    "k2do/providers/__init__.py",
    "k2do/providers/base.py",
    "k2do/labs/deepthink_trace.py",
    "k2do/labs/trace_evidence_recorder.py",
)
_ARTIFACT_MEDIA_TYPES: Final = {
    "receipt.json": "application/json",
    "trace-lab.txt": "text/plain; charset=utf-8",
    "trace-lab.svg": "image/svg+xml",
    "call-dag.svg": "image/svg+xml",
    "orchestration-properties.svg": "image/svg+xml",
}
_COMMUNICATION_SYSCALLS: Final = (
    "socket",
    "bind",
    "listen",
    "connect",
    "accept",
    "accept4",
    "sendto",
    "sendmsg",
    "recvfrom",
    "recvmsg",
)
_NETWORK_OBSERVATION_SCOPE: Final = (
    "one Linux strace run of the offline lab and all child threads; "
    "local event-loop socketpair creation is outside this communication-syscall set"
)
_STRACE_PATH: Final = "/usr/bin/strace"
_ISOLATED_MODULE: Final = "k2do.labs.deepthink_trace"
_ISOLATED_BOOTSTRAP: Final = (
    "import runpy,sys;"
    "snapshot=sys.argv[1];"
    "dependencies=sys.argv[2:];"
    "sys.path[:0]=[snapshot,*dependencies];"
    f"sys.argv=['{_ISOLATED_MODULE}'];"
    f"runpy.run_module('{_ISOLATED_MODULE}',run_name='__main__')"
)
_READER_COMMAND: Final = ("python", "-m", _ISOLATED_MODULE)
_TRANSCRIPT_COMMAND: Final = "$ python -m k2do.labs.deepthink_trace\n"
_HEX_OBJECT_RE: Final = re.compile(r"[0-9a-f]{40,64}")
_HEX_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_SAFE_RUNTIME_RE: Final = re.compile(r"[A-Za-z0-9_.+() -]{1,120}")


class EvidenceError(RuntimeError):
    """A fail-closed recorder or verifier error with a public categorical code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


CaptureFunction = Callable[[Path], tuple[bytes, dict[str, Any]]]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            env=_safe_git_environment(),
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("git_invocation_failed") from exc
    _require(completed.returncode == 0, "git_contract_failed")
    _require(
        len(completed.stdout) <= _MAX_COMMAND_BYTES and len(completed.stderr) <= _MAX_COMMAND_BYTES,
        "git_output_exceeded_limit",
    )
    return completed.stdout


def _git_object_exists(repository: Path, object_specification: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", object_specification],
            cwd=repository,
            env=_safe_git_environment(),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("git_invocation_failed") from exc
    _require(
        len(completed.stdout) <= 4096 and len(completed.stderr) <= 4096,
        "git_output_exceeded_limit",
    )
    if completed.returncode == 0:
        return True
    _require(completed.returncode in {1, 128}, "git_contract_failed")
    return False


def _assert_clean_committed_head(repository: Path) -> tuple[str, str]:
    head = _run_git(repository, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    tree = _run_git(repository, ["rev-parse", "--verify", "HEAD^{tree}"]).decode().strip()
    _require(bool(_HEX_OBJECT_RE.fullmatch(head)), "head_identity_invalid")
    _require(bool(_HEX_OBJECT_RE.fullmatch(tree)), "tree_identity_invalid")
    status_output = _run_git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    _require(status_output == b"", "record_requires_clean_tree")
    return head, tree


def _assert_publication_state(
    repository: Path,
    *,
    expected_head: str,
    expected_tree: str,
    expected_sources: Mapping[str, Mapping[str, Any]],
    stage_name: str,
) -> None:
    head = _run_git(repository, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    tree = _run_git(repository, ["rev-parse", "--verify", "HEAD^{tree}"]).decode().strip()
    _require(
        head == expected_head
        and tree == expected_tree
        and _source_inventory(repository) == expected_sources,
        "source_changed_before_publication",
    )
    status_output = _run_git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    expected_names = sorted(set(_ARTIFACT_MEDIA_TYPES) | {"manifest.json"})
    expected_status = b"".join(
        f"?? docs/{stage_name}/{name}\0".encode("utf-8") for name in expected_names
    )
    _require(status_output == expected_status, "worktree_changed_before_publication")


def _open_repository_root(repository: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(repository, flags)
    except OSError as exc:
        raise EvidenceError("repository_root_not_safe") from exc


def _read_relative_regular(
    root_fd: int,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    path = PurePosixPath(relative_path)
    _require(
        not path.is_absolute()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts),
        "relative_path_invalid",
    )
    directory_fd = os.dup(root_fd)
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        for component in path.parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise EvidenceError("source_directory_not_safe") from exc
            os.close(directory_fd)
            directory_fd = next_fd

        file_flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            file_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        try:
            file_fd = os.open(path.parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise EvidenceError("regular_file_open_failed") from exc
        try:
            metadata = os.fstat(file_fd)
            _require(stat.S_ISREG(metadata.st_mode), "artifact_not_regular")
            _require(metadata.st_size <= maximum_bytes, "artifact_exceeded_limit")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining > 0:
                chunk = os.read(file_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            _require(len(data) <= maximum_bytes, "artifact_exceeded_limit")
            _require(len(data) == metadata.st_size, "artifact_size_changed_during_read")
            return data
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def _tree_entry(
    repository: Path,
    treeish: str,
    relative_path: str,
) -> tuple[str, str]:
    output = _run_git(
        repository,
        ["ls-tree", "-z", treeish, "--", relative_path],
    )
    records = [record for record in output.split(b"\0") if record]
    _require(len(records) == 1, "source_tree_entry_missing")
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        tree_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError("source_tree_entry_invalid") from exc
    _require(
        tree_path == relative_path
        and object_type == "blob"
        and mode in {"100644", "100755"}
        and bool(_HEX_OBJECT_RE.fullmatch(object_id)),
        "source_tree_entry_invalid",
    )
    return mode, object_id


def _head_tree_entry(repository: Path, relative_path: str) -> tuple[str, str]:
    return _tree_entry(repository, "HEAD", relative_path)


def _source_inventory(repository: Path) -> dict[str, dict[str, Any]]:
    root_fd = _open_repository_root(repository)
    try:
        inventory: dict[str, dict[str, Any]] = {}
        for relative_path in _SOURCE_PATHS:
            worktree_data = _read_relative_regular(
                root_fd,
                relative_path,
                maximum_bytes=_MAX_ARTIFACT_BYTES,
            )
            git_mode, head_blob = _head_tree_entry(repository, relative_path)
            committed_data = _run_git(
                repository,
                ["cat-file", "blob", head_blob],
            )
            _require(
                worktree_data == committed_data,
                "source_worktree_content_mismatch",
            )
            inventory[relative_path] = {
                "bytes": len(committed_data),
                "git_blob": head_blob,
                "git_mode": git_mode,
                "sha256": _sha256(committed_data),
            }
        return inventory
    finally:
        os.close(root_fd)


def _committed_package_entries(
    repository: Path,
    head: str,
) -> list[tuple[str, str, str]]:
    output = _run_git(
        repository,
        ["ls-tree", "-rz", "--full-tree", head, "--", "k2do"],
    )
    entries: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()
    for record in (item for item in output.split(b"\0") if item):
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            relative_path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise EvidenceError("snapshot_tree_entry_invalid") from exc
        path = PurePosixPath(relative_path)
        _require(
            object_type == "blob"
            and mode in {"100644", "100755"}
            and bool(_HEX_OBJECT_RE.fullmatch(object_id))
            and path.parts
            and path.parts[0] == "k2do"
            and all(part not in {"", ".", ".."} for part in path.parts)
            and relative_path not in seen_paths,
            "snapshot_tree_entry_invalid",
        )
        seen_paths.add(relative_path)
        entries.append((relative_path, mode, object_id))
    _require(entries and entries == sorted(entries), "snapshot_tree_not_canonical")
    return entries


def _open_child_directory(directory_fd: int, component: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(component, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise EvidenceError("snapshot_directory_not_safe") from exc


def _ensure_snapshot_parent(
    snapshot_fd: int,
    parent_parts: Sequence[str],
    created_directories: dict[tuple[str, ...], tuple[int, int]],
) -> int:
    directory_fd = os.dup(snapshot_fd)
    current: list[str] = []
    try:
        for component in parent_parts:
            current.append(component)
            key = tuple(current)
            if key not in created_directories:
                try:
                    os.mkdir(component, 0o700, dir_fd=directory_fd)
                except OSError as exc:
                    raise EvidenceError("snapshot_directory_create_failed") from exc
            next_fd = _open_child_directory(directory_fd, component)
            metadata = os.fstat(next_fd)
            identity = (metadata.st_dev, metadata.st_ino)
            if key in created_directories:
                _require(
                    identity == created_directories[key],
                    "snapshot_directory_identity_changed",
                )
            else:
                created_directories[key] = identity
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _open_snapshot_parent(snapshot_fd: int, parent_parts: Sequence[str]) -> int:
    directory_fd = os.dup(snapshot_fd)
    try:
        for component in parent_parts:
            next_fd = _open_child_directory(directory_fd, component)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _unlink_snapshot_file(
    snapshot_fd: int,
    parts: Sequence[str],
    identity: tuple[int, int],
) -> None:
    parent_fd = _open_snapshot_parent(snapshot_fd, parts[:-1])
    try:
        metadata = os.stat(
            parts[-1],
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require(
            stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == identity,
            "snapshot_file_identity_changed",
        )
        os.unlink(parts[-1], dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _remove_snapshot_directory(
    snapshot_fd: int,
    parts: Sequence[str],
    identity: tuple[int, int],
) -> None:
    parent_fd = _open_snapshot_parent(snapshot_fd, parts[:-1])
    try:
        metadata = os.stat(
            parts[-1],
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require(
            stat.S_ISDIR(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == identity,
            "snapshot_directory_identity_changed",
        )
        os.rmdir(parts[-1], dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


@contextmanager
def _committed_snapshot(repository: Path, head: str) -> Iterator[Path]:
    entries = _committed_package_entries(repository, head)
    root_fd = _open_repository_root(repository)
    snapshot_fd = -1
    snapshot_name = f".trace-evidence-snapshot-{os.getpid()}-{secrets.token_hex(8)}"
    snapshot_identity: tuple[int, int] | None = None
    created_files: list[tuple[tuple[str, ...], tuple[int, int]]] = []
    created_directories: dict[tuple[str, ...], tuple[int, int]] = {}
    try:
        try:
            os.mkdir(snapshot_name, 0o700, dir_fd=root_fd)
        except OSError as exc:
            raise EvidenceError("snapshot_root_create_failed") from exc
        snapshot_fd = _open_child_directory(root_fd, snapshot_name)
        root_metadata = os.fstat(snapshot_fd)
        _require(stat.S_ISDIR(root_metadata.st_mode), "snapshot_root_not_directory")
        snapshot_identity = (root_metadata.st_dev, root_metadata.st_ino)

        for relative_path, git_mode, object_id in entries:
            parts = PurePosixPath(relative_path).parts
            parent_fd = _ensure_snapshot_parent(
                snapshot_fd,
                parts[:-1],
                created_directories,
            )
            try:
                data = _run_git(repository, ["cat-file", "blob", object_id])
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                file_mode = 0o700 if git_mode == "100755" else 0o600
                try:
                    file_fd = os.open(
                        parts[-1],
                        flags,
                        file_mode,
                        dir_fd=parent_fd,
                    )
                except OSError as exc:
                    raise EvidenceError("snapshot_file_create_failed") from exc
                try:
                    file_metadata = os.fstat(file_fd)
                    created_files.append(
                        (
                            tuple(parts),
                            (file_metadata.st_dev, file_metadata.st_ino),
                        )
                    )
                    _write_all(file_fd, data)
                finally:
                    os.close(file_fd)
            finally:
                os.close(parent_fd)
        yield repository / snapshot_name
    finally:
        cleanup_error = False
        if snapshot_fd >= 0:
            for parts, identity in reversed(created_files):
                try:
                    _unlink_snapshot_file(snapshot_fd, parts, identity)
                except (OSError, EvidenceError):
                    cleanup_error = True
            for parts, identity in sorted(
                created_directories.items(),
                key=lambda item: (len(item[0]), item[0]),
                reverse=True,
            ):
                try:
                    _remove_snapshot_directory(snapshot_fd, parts, identity)
                except (OSError, EvidenceError):
                    cleanup_error = True
            os.close(snapshot_fd)
        if snapshot_identity is not None:
            try:
                path_metadata = os.stat(
                    snapshot_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(path_metadata.st_mode)
                    or (path_metadata.st_dev, path_metadata.st_ino) != snapshot_identity
                ):
                    cleanup_error = True
                else:
                    os.rmdir(snapshot_name, dir_fd=root_fd)
            except OSError:
                cleanup_error = True
        os.close(root_fd)
        if cleanup_error and sys.exc_info()[0] is None:
            raise EvidenceError("snapshot_cleanup_failed")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    else:
        time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()


def _run_bounded(
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_s: float,
    stdout_limit: int = _MAX_COMMAND_BYTES,
    stderr_limit: int = _MAX_COMMAND_BYTES,
) -> tuple[int, bytes, bytes]:
    try:
        process = subprocess.Popen(
            list(arguments),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise EvidenceError("capture_process_start_failed") from exc
    _require(process.stdout is not None and process.stderr is not None, "capture_pipe_missing")

    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    os.set_blocking(stdout_fd, False)
    os.set_blocking(stderr_fd, False)
    selector = selectors.DefaultSelector()
    selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
    selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    totals = {"stdout": 0, "stderr": 0}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    deadline = time.monotonic() + timeout_s
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                process.wait()
                raise EvidenceError("capture_timed_out")
            for key, _ in selector.select(min(remaining, 0.25)):
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                stream = key.data
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                totals[stream] += len(chunk)
                if totals[stream] > limits[stream]:
                    _terminate_process_group(process)
                    process.wait()
                    raise EvidenceError("capture_output_exceeded_limit")
                chunks[stream].append(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            process.wait()
            raise EvidenceError("capture_timed_out")
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            process.wait()
            raise EvidenceError("capture_timed_out") from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()

    return return_code, b"".join(chunks["stdout"]), b"".join(chunks["stderr"])


def _capture_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }


def _strace_version(strace_path: str, repository: Path) -> str:
    return_code, stdout, stderr = _run_bounded(
        [strace_path, "--version"],
        cwd=repository,
        environment=_capture_environment(),
        timeout_s=5,
        stdout_limit=4096,
        stderr_limit=4096,
    )
    _require(return_code == 0 and stderr == b"", "strace_version_failed")
    try:
        first_line = stdout.decode("ascii").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise EvidenceError("strace_version_invalid") from exc
    _require(bool(_SAFE_RUNTIME_RE.fullmatch(first_line)), "strace_version_invalid")
    return first_line


def _json_without_duplicate_keys(data: bytes, code: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(code)
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=reject_duplicates)
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(code) from exc


def _validate_captured_receipt(receipt_bytes: bytes) -> dict[str, Any]:
    _require(
        receipt_bytes.endswith(b"\n") and len(receipt_bytes) <= _MAX_COMMAND_BYTES,
        "receipt_encoding_invalid",
    )
    receipt = _json_without_duplicate_keys(receipt_bytes, "receipt_json_invalid")
    _require(isinstance(receipt, dict), "receipt_root_invalid")
    try:
        rendered = deepthink_trace.render_receipt(receipt).encode("utf-8")
    except Exception as exc:
        raise EvidenceError("receipt_contract_invalid") from exc
    _require(rendered == receipt_bytes, "receipt_not_canonical")
    return receipt


def _dependency_import_paths() -> list[str]:
    paths: list[str] = []
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(key)
        if candidate and candidate not in paths:
            resolved = os.path.realpath(candidate)
            _require(os.path.isabs(resolved) and os.path.isdir(resolved), "dependency_path_invalid")
            paths.append(resolved)
    _require(bool(paths), "dependency_path_missing")
    return paths


def _validate_strace_binary() -> str:
    try:
        metadata = os.stat(_STRACE_PATH, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceError("strace_not_available") from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and metadata.st_uid == 0 and metadata.st_mode & 0o022 == 0,
        "strace_binary_not_trusted",
    )
    return _STRACE_PATH


def _capture_from_snapshot(
    repository: Path,
    snapshot: Path,
) -> tuple[bytes, dict[str, Any]]:
    _require(platform.system() == "Linux", "strace_capture_requires_linux")
    strace_path = _validate_strace_binary()
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _ISOLATED_BOOTSTRAP,
        str(snapshot),
        *_dependency_import_paths(),
    ]
    environment = _capture_environment()

    direct_code, direct_stdout, direct_stderr = _run_bounded(
        command,
        cwd=snapshot,
        environment=environment,
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    _require(direct_code == 0 and direct_stderr == b"", "trace_lab_capture_failed")
    _validate_captured_receipt(direct_stdout)

    trace_expression = ",".join(_COMMUNICATION_SYSCALLS)
    traced_code, traced_stdout, traced_stderr = _run_bounded(
        [
            strace_path,
            "-f",
            "-qq",
            "-e",
            f"trace={trace_expression}",
            "--",
            *command,
        ],
        cwd=snapshot,
        environment=environment,
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    _require(traced_code == 0, "strace_capture_failed")
    _require(traced_stdout == direct_stdout, "traced_receipt_changed")
    _require(traced_stderr == b"", "communication_syscall_observed")

    observation = {
        "communication_syscall_count": 0,
        "scope": _NETWORK_OBSERVATION_SCOPE,
        "status": "no_communication_syscalls_observed",
        "syscalls": list(_COMMUNICATION_SYSCALLS),
        "tool": _strace_version(strace_path, repository),
    }
    return direct_stdout, observation


def _capture_receipt(repository: Path) -> tuple[bytes, dict[str, Any]]:
    head = _run_git(repository, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    _require(bool(_HEX_OBJECT_RE.fullmatch(head)), "head_identity_invalid")
    with _committed_snapshot(repository, head) as snapshot:
        return _capture_from_snapshot(repository, snapshot)


def _scenario(receipt: Mapping[str, Any], identifier: str) -> Mapping[str, Any]:
    scenarios = receipt.get("scenarios")
    _require(isinstance(scenarios, list), "receipt_scenarios_invalid")
    matches = [
        item for item in scenarios if isinstance(item, dict) and item.get("id") == identifier
    ]
    _require(len(matches) == 1, "receipt_scenario_missing")
    return matches[0]


def _xml_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_document(
    *,
    width: int,
    height: int,
    title: str,
    description: str,
    body: str,
) -> bytes:
    title_id = "visual-title"
    description_id = "visual-description"
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {description_id}">
  <title id="{title_id}">{_xml_text(title)}</title>
  <desc id="{description_id}">{_xml_text(description)}</desc>
  <defs>
    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#08111f"/>
      <stop offset="100%" stop-color="#10233f"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#020617" flood-opacity=".42"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7dd3fc"/>
    </marker>
  </defs>
  <rect width="{width}" height="{height}" rx="24" fill="url(#background)"/>
  <style>
    .sans {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .mono {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }}
  </style>
{body}
</svg>
"""
    return document.encode("utf-8")


def _validate_svg(svg_bytes: bytes) -> None:
    lowered = svg_bytes.lower()
    for forbidden in (
        b"<!doctype",
        b"<!entity",
        b"<script",
        b"<foreignobject",
        b"<metadata",
        b" href=",
        b"xlink:href",
        b"javascript:",
        b"data:",
        b"url(http",
        b"url(//",
    ):
        _require(forbidden not in lowered, "svg_active_content_forbidden")
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise EvidenceError("svg_xml_invalid") from exc
    _require(
        root.tag == "{http://www.w3.org/2000/svg}svg" and root.attrib.get("role") == "img",
        "svg_root_invalid",
    )


def _render_terminal_svg(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    network_observation: Mapping[str, Any],
) -> bytes:
    router = receipt["router"]
    cases = router["cases"]
    synthesis = _scenario(receipt, "synthesis_with_thinker_fallback_and_timeout")
    degradation = _scenario(
        receipt,
        "judge_failure_uses_longest_successful_thinker",
    )
    cancellation = _scenario(
        receipt,
        "caller_cancellation_drains_parallel_thinkers",
    )
    lines = [
        (_TRANSCRIPT_COMMAND.rstrip("\n"), "#e2e8f0"),
        (f"schema                 {receipt['schema']}", "#94a3b8"),
        (f"status                 {receipt['status']}", "#86efac"),
        (f"router complex/simple  {cases[0]['route']} / {cases[1]['route']}", "#7dd3fc"),
        (
            "synthesis             "
            f"peak={synthesis['concurrency']['provider_peak_calls']}  "
            f"fallback={' -> '.join(synthesis['fallback']['path'])}",
            "#f8fafc",
        ),
        (
            "thinker timeout       "
            f"{synthesis['timeout']['actor']} / {synthesis['timeout']['result']}",
            "#fbbf24",
        ),
        (
            "judge degradation     "
            f"{' -> '.join(degradation['judge']['model_path'])} / "
            f"{degradation['judge']['mode']}",
            "#f8fafc",
        ),
        (
            "caller cancellation   "
            f"cancelled={cancellation['cancellation']['cancelled_provider_calls']}  "
            f"judge_calls={cancellation['cancellation']['judge_calls']}  "
            f"active_after={cancellation['cleanup']['active_provider_calls']}",
            "#f8fafc",
        ),
        (
            "network observation   "
            f"{network_observation['communication_syscall_count']} observed in listed strace set",
            "#86efac",
        ),
        (f"receipt sha256         {receipt_sha256}", "#94a3b8"),
    ]
    rendered_lines = []
    for index, (line, color) in enumerate(lines):
        rendered_lines.append(
            f'    <text x="82" y="{132 + index * 46}" class="mono" '
            f'font-size="20" fill="{color}">{_xml_text(line)}</text>'
        )
    rendered_markup = "\n".join(rendered_lines)
    body = f"""  <g filter="url(#shadow)">
    <rect x="48" y="48" width="1104" height="624" rx="18" fill="#020617" stroke="#334155"/>
    <rect x="48" y="48" width="1104" height="54" rx="18" fill="#111c2f"/>
    <rect x="48" y="84" width="1104" height="18" fill="#111c2f"/>
    <circle cx="82" cy="75" r="7" fill="#fb7185"/>
    <circle cx="106" cy="75" r="7" fill="#fbbf24"/>
    <circle cx="130" cy="75" r="7" fill="#4ade80"/>
    <text x="600" y="82" text-anchor="middle" class="sans" font-size="16" fill="#94a3b8">offline trace capture · no credentials · no live provider</text>
{rendered_markup}
  </g>"""
    return _svg_document(
        width=1200,
        height=720,
        title="K2DO offline DeepThink trace capture",
        description=(
            "A terminal rendering derived from the canonical credential-free trace receipt. "
            "It shows routing, fallback, timeout, judge degradation, cancellation cleanup, "
            "the communication-syscall observation, and the receipt digest."
        ),
        body=body,
    )


def _render_call_dag_svg(receipt: Mapping[str, Any]) -> bytes:
    synthesis = _scenario(receipt, "synthesis_with_thinker_fallback_and_timeout")
    call_dag = synthesis["call_dag"]
    nodes = call_dag["nodes"]
    edges = call_dag["edges"]
    positions = {
        "router": (100, 340),
        "thinker.analyst.primary": (320, 200),
        "thinker.analyst.fallback": (590, 200),
        "thinker.creative.primary": (320, 340),
        "thinker.pragmatist.primary": (320, 500),
        "judge.primary": (825, 340),
        "result": (1080, 340),
    }
    node_width = 156
    node_height = 90
    node_half_width = node_width // 2
    node_half_height = node_height // 2
    content_top = 145
    footer_top = 585
    _require(
        {node["id"] for node in nodes} == set(positions),
        "synthesis_dag_shape_changed",
    )
    _require(
        min(y - node_half_height for _, y in positions.values()) >= content_top,
        "dag_node_overlaps_header",
    )
    _require(
        max(y + node_half_height for _, y in positions.values()) < footer_top,
        "dag_node_overlaps_footer",
    )
    node_by_id = {node["id"]: node for node in nodes}
    edge_markup: list[str] = []
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        _require(source in positions and target in positions, "synthesis_edge_unknown")
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        direction = 1 if target_x >= source_x else -1
        start_x = source_x + node_half_width * direction
        end_x = target_x - node_half_width * direction
        mid_x = (start_x + end_x) / 2
        mid_y = (source_y + target_y) / 2
        relation = str(edge["relation"])
        label_width = len(relation) * 6 + 10
        _require(
            label_width <= abs(end_x - start_x),
            "dag_edge_label_does_not_fit",
        )
        _require(mid_y - 22 >= content_top, "dag_edge_label_overlaps_header")
        edge_markup.append(
            f'    <path d="M {start_x} {source_y} C {mid_x} {source_y}, '
            f'{mid_x} {target_y}, {end_x} {target_y}" fill="none" '
            'stroke="#7dd3fc" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )
        edge_markup.append(
            f'    <rect x="{mid_x - label_width / 2}" y="{mid_y - 22}" '
            f'width="{label_width}" height="18" rx="7" fill="#08111f" '
            'stroke="#334155"/>'
        )
        edge_markup.append(
            f'    <text x="{mid_x}" y="{mid_y - 9}" text-anchor="middle" '
            'class="mono" font-size="10" fill="#cbd5e1">'
            f"{_xml_text(relation)}</text>"
        )

    palette = {
        "deepthink": ("#0c4a6e", "#7dd3fc"),
        "error": ("#4c1d1d", "#fca5a5"),
        "success": ("#14532d", "#86efac"),
        "timeout": ("#713f12", "#fde68a"),
        "synthesis": ("#312e81", "#c4b5fd"),
    }
    node_markup: list[str] = []
    for node_id, (x, y) in positions.items():
        node = node_by_id[node_id]
        fill, accent = palette.get(node["outcome"], ("#1e293b", "#e2e8f0"))
        display = node_id.replace("thinker.", "").replace(".", " · ")
        node_markup.append(
            f"""    <g transform="translate({x - node_half_width} {y - node_half_height})" filter="url(#shadow)">
      <rect width="{node_width}" height="{node_height}" rx="14" fill="{fill}" stroke="{accent}" stroke-opacity=".7"/>
      <text x="{node_half_width}" y="36" text-anchor="middle" class="sans" font-size="13" font-weight="700" fill="#f8fafc">{_xml_text(display)}</text>
      <text x="{node_half_width}" y="64" text-anchor="middle" class="mono" font-size="12" fill="{accent}">{_xml_text(node["outcome"])}</text>
    </g>"""
        )
    edges_rendered = "\n".join(edge_markup)
    nodes_rendered = "\n".join(node_markup)
    body = f"""  <text x="56" y="58" class="sans" font-size="30" font-weight="750" fill="#f8fafc">Receipt-derived synthesis call DAG</text>
  <text x="56" y="88" class="sans" font-size="16" fill="#94a3b8">Production router + DeepThinkEngine · strict scripted provider · no quality or latency claim</text>
  <g>
{edges_rendered}
{nodes_rendered}
  </g>
  <g transform="translate(56 {footer_top})">
    <rect width="1088" height="54" rx="12" fill="#0f1d31" stroke="#334155"/>
    <text x="22" y="33" class="sans" font-size="15" fill="#cbd5e1">Judge gate opens only after every selected thinker is terminal; the timeout is a categorical engine path, not a benchmark.</text>
  </g>"""
    return _svg_document(
        width=1200,
        height=680,
        title="K2DO synthesis call DAG",
        description=(
            "The synthesis scenario call DAG rendered from receipt nodes and edges, "
            "including thinker fallback, timeout, judge gating, and synthesis."
        ),
        body=body,
    )


def _render_properties_svg(receipt: Mapping[str, Any]) -> bytes:
    scenarios = [
        _scenario(receipt, "synthesis_with_thinker_fallback_and_timeout"),
        _scenario(receipt, "judge_failure_uses_longest_successful_thinker"),
        _scenario(receipt, "caller_cancellation_drains_parallel_thinkers"),
    ]
    titles = ("Synthesis path", "Judge degradation", "Caller cancellation")
    accents = ("#38bdf8", "#a78bfa", "#fb7185")
    cards: list[str] = []
    for index, (scenario, title, accent) in enumerate(zip(scenarios, titles, accents)):
        x = 44 + index * 382
        peak = scenario["concurrency"]["provider_peak_calls"]
        maximum = scenario["concurrency"]["configured_max_agents"]
        cleanup = scenario["cleanup"]["active_provider_calls"]
        if index == 0:
            detail_lines = (
                ("THINKER FALLBACK", "label"),
                (scenario["fallback"]["path"][0], "value"),
                (f"-> {scenario['fallback']['path'][1]}", "value"),
                ("TIMEOUT", "label"),
                (
                    f"{scenario['timeout']['actor']} -> {scenario['timeout']['result']}",
                    "value",
                ),
                (
                    "provider_cancelled="
                    f"{str(scenario['timeout']['provider_call_cancelled']).lower()}",
                    "value",
                ),
                (f"JUDGE GATE · ACTIVE_AFTER={cleanup}", "label"),
                (scenario["judge"]["gate"], "value"),
            )
        elif index == 1:
            detail_lines = (
                ("JUDGE MODEL PATH", "label"),
                (scenario["judge"]["model_path"][0], "value"),
                (f"-> {scenario['judge']['model_path'][1]}", "value"),
                ("SELECTION", "label"),
                (scenario["judge"]["mode"], "value"),
                ("SELECTED", "label"),
                (scenario["judge"]["selected"], "value"),
                (f"ACTIVE_AFTER={cleanup}", "label"),
            )
        else:
            detail_lines = (
                ("CANCELLATION", "label"),
                (
                    f"blocked={scenario['cancellation']['blocked_provider_calls']}",
                    "value",
                ),
                (
                    f"cancelled={scenario['cancellation']['cancelled_provider_calls']}",
                    "value",
                ),
                ("PROPAGATION", "label"),
                (scenario["cancellation"]["propagation"], "value"),
                (f"judge_calls={scenario['cancellation']['judge_calls']}", "value"),
                ("ROOT · CLEANUP", "label"),
                (
                    f"{scenario['cleanup']['root_task']} · active_after={cleanup}",
                    "value",
                ),
            )
        _require(
            len(str(scenario["id"])) <= 45
            and all(len(str(detail)) <= 40 for detail, _kind in detail_lines),
            "properties_text_does_not_fit",
        )
        detail_markup = "\n".join(
            f'      <text x="24" y="{264 + row * 20}" class="mono" '
            f'font-size="{"11" if kind == "label" else "12"}" '
            f'font-weight="{"700" if kind == "label" else "400"}" '
            f'fill="{"#64748b" if kind == "label" else "#cbd5e1"}">'
            f"{_xml_text(detail)}</text>"
            for row, (detail, kind) in enumerate(detail_lines)
        )
        bar_width = int(286 * peak / max(1, maximum))
        cards.append(
            f"""  <g transform="translate({x} 130)" filter="url(#shadow)">
    <rect width="348" height="438" rx="18" fill="#0b1729" stroke="#334155"/>
    <rect width="348" height="7" rx="4" fill="{accent}"/>
    <text x="24" y="52" class="sans" font-size="21" font-weight="700" fill="#f8fafc">{_xml_text(title)}</text>
    <text x="24" y="86" class="mono" font-size="12" fill="#64748b">{_xml_text(scenario["id"])}</text>
    <text x="24" y="145" class="sans" font-size="14" fill="#94a3b8">observed provider peak</text>
    <text x="24" y="203" class="sans" font-size="54" font-weight="800" fill="{accent}">{peak}</text>
    <text x="70" y="201" class="sans" font-size="16" fill="#94a3b8">of configured max {maximum}</text>
    <rect x="24" y="220" width="286" height="10" rx="5" fill="#1e293b"/>
    <rect x="24" y="220" width="{bar_width}" height="10" rx="5" fill="{accent}"/>
{detail_markup}
  </g>"""
        )
    cards_rendered = "\n".join(cards)
    body = f"""  <text x="44" y="58" class="sans" font-size="30" font-weight="750" fill="#f8fafc">Observed orchestration properties</text>
  <text x="44" y="90" class="sans" font-size="16" fill="#94a3b8">Three deterministic offline scenarios · categorical control-flow evidence · timings excluded</text>
{cards_rendered}
  <text x="600" y="626" text-anchor="middle" class="sans" font-size="15" fill="#94a3b8">Bars encode observed concurrent provider calls against each configured max; they do not encode speed or throughput.</text>"""
    return _svg_document(
        width=1200,
        height=660,
        title="K2DO observed orchestration properties",
        description=(
            "Three receipt-derived panels compare concurrency, fallback, timeout, "
            "judge degradation, cancellation, and cleanup outcomes."
        ),
        body=body,
    )


def _render_transcript(receipt_bytes: bytes) -> bytes:
    return _TRANSCRIPT_COMMAND.encode("utf-8") + receipt_bytes


def _render_artifacts(
    receipt_bytes: bytes,
    network_observation: Mapping[str, Any],
) -> dict[str, bytes]:
    receipt = _validate_captured_receipt(receipt_bytes)
    receipt_sha256 = _sha256(receipt_bytes)
    artifacts = {
        "receipt.json": receipt_bytes,
        "trace-lab.txt": _render_transcript(receipt_bytes),
        "trace-lab.svg": _render_terminal_svg(
            receipt,
            receipt_sha256=receipt_sha256,
            network_observation=network_observation,
        ),
        "call-dag.svg": _render_call_dag_svg(receipt),
        "orchestration-properties.svg": _render_properties_svg(receipt),
    }
    _require(set(artifacts) == set(_ARTIFACT_MEDIA_TYPES), "artifact_set_changed")
    for data in artifacts.values():
        _require(len(data) <= _MAX_ARTIFACT_BYTES, "artifact_exceeded_limit")
    for name in ("trace-lab.svg", "call-dag.svg", "orchestration-properties.svg"):
        _validate_svg(artifacts[name])
    return artifacts


def _runtime_manifest() -> dict[str, Any]:
    try:
        k2do_version = importlib.metadata.version("k2do")
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvidenceError("k2do_distribution_missing") from exc
    values = (
        platform.python_implementation(),
        platform.python_version(),
        platform.system(),
        platform.machine(),
        k2do_version,
    )
    _require(
        all(bool(_SAFE_RUNTIME_RE.fullmatch(value)) for value in values),
        "runtime_metadata_invalid",
    )
    return {
        "architecture": platform.machine(),
        "dependency_environment_locked": True,
        "environment_claim": (
            "CPython 3.12 CI dependencies are hash-locked by requirements-ci-py312.lock"
        ),
        "k2do_distribution_version": k2do_version,
        "operating_system": platform.system(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def _build_manifest(
    *,
    head: str,
    tree: str,
    sources: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, bytes],
    network_observation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifacts": {
            name: {
                "bytes": len(artifacts[name]),
                "media_type": _ARTIFACT_MEDIA_TYPES[name],
                "sha256": _sha256(artifacts[name]),
            }
            for name in sorted(artifacts)
        },
        "capture": {
            "execution": {
                "bootstrap_sha256": _sha256(_ISOLATED_BOOTSTRAP.encode("utf-8")),
                "dependency_imports": (
                    "plain interpreter purelib/platlib paths; absolute paths excluded"
                ),
                "module": _ISOLATED_MODULE,
                "python_flags": ["-I", "-S", "-B"],
                "snapshot": ("private materialization of committed HEAD; absolute path excluded"),
            },
            "network_observation": dict(network_observation),
            "reader_command": list(_READER_COMMAND),
            "receipt_sha256": _sha256(artifacts["receipt.json"]),
            "runtime": _runtime_manifest(),
            "stdout_bytes": len(artifacts["receipt.json"]),
        },
        "schema": _MANIFEST_SCHEMA,
        "source": {
            "commit": head,
            "files": {name: dict(sources[name]) for name in sorted(sources)},
            "tree": tree,
            "verification": (
                "current committed blob identities and content hashes; no ancestry requirement"
            ),
        },
    }


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    target_directory_fd: int,
    target_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2_not_available")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        target_directory_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise EvidenceError("evidence_directory_already_exists")
        raise EvidenceError("atomic_publication_failed")


def _write_all(file_fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_fd, view)
        _require(written > 0, "artifact_write_failed")
        view = view[written:]


def _publish_artifacts(
    repository: Path,
    files: Mapping[str, bytes],
    *,
    pre_publish: Callable[[str], None] | None = None,
) -> None:
    expected_names = set(_ARTIFACT_MEDIA_TYPES) | {"manifest.json"}
    _require(set(files) == expected_names, "publication_file_set_changed")
    root_fd = _open_repository_root(repository)
    docs_fd = -1
    stage_fd = -1
    stage_name: str | None = None
    stage_identity: tuple[int, int] | None = None
    written_files: dict[str, tuple[int, int]] = {}
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            docs_fd = os.open("docs", directory_flags, dir_fd=root_fd)
        except OSError as exc:
            raise EvidenceError("docs_directory_not_safe") from exc
        try:
            os.stat(_OUTPUT_DIRECTORY, dir_fd=docs_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("evidence_directory_already_exists")

        stage_name = f".{_OUTPUT_DIRECTORY}.stage-{os.getpid()}-{secrets.token_hex(8)}"
        os.mkdir(stage_name, 0o700, dir_fd=docs_fd)
        stage_fd = os.open(stage_name, directory_flags, dir_fd=docs_fd)
        stage_metadata = os.fstat(stage_fd)
        _require(stat.S_ISDIR(stage_metadata.st_mode), "staging_leaf_not_directory")
        stage_identity = (stage_metadata.st_dev, stage_metadata.st_ino)
        for name in sorted(files):
            _require("/" not in name and name not in {".", ".."}, "artifact_name_invalid")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            file_fd = os.open(name, flags, 0o600, dir_fd=stage_fd)
            try:
                file_metadata = os.fstat(file_fd)
                written_files[name] = (
                    file_metadata.st_dev,
                    file_metadata.st_ino,
                )
                os.fchmod(file_fd, 0o600)
                _write_all(file_fd, files[name])
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
        os.fsync(stage_fd)
        if pre_publish is not None:
            pre_publish(stage_name)
        os.close(stage_fd)
        stage_fd = -1
        _rename_noreplace(docs_fd, stage_name, docs_fd, _OUTPUT_DIRECTORY)
        stage_name = None
        os.fsync(docs_fd)
    except OSError as exc:
        raise EvidenceError("artifact_publication_io_failed") from exc
    finally:
        cleanup_fd = stage_fd
        if (
            cleanup_fd < 0
            and stage_name is not None
            and stage_identity is not None
            and docs_fd >= 0
        ):
            try:
                path_metadata = os.stat(
                    stage_name,
                    dir_fd=docs_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(path_metadata.st_mode)
                    and (path_metadata.st_dev, path_metadata.st_ino) == stage_identity
                ):
                    cleanup_fd = os.open(
                        stage_name,
                        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=docs_fd,
                    )
            except OSError:
                pass
        if cleanup_fd >= 0:
            try:
                cleanup_metadata = os.fstat(cleanup_fd)
                if (
                    stage_identity is not None
                    and (cleanup_metadata.st_dev, cleanup_metadata.st_ino) == stage_identity
                ):
                    for name, identity in written_files.items():
                        try:
                            file_metadata = os.stat(
                                name,
                                dir_fd=cleanup_fd,
                                follow_symlinks=False,
                            )
                            if (
                                stat.S_ISREG(file_metadata.st_mode)
                                and (file_metadata.st_dev, file_metadata.st_ino) == identity
                            ):
                                os.unlink(name, dir_fd=cleanup_fd)
                        except OSError:
                            pass
            finally:
                os.close(cleanup_fd)
        if stage_name is not None and stage_identity is not None and docs_fd >= 0:
            try:
                path_metadata = os.stat(
                    stage_name,
                    dir_fd=docs_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(path_metadata.st_mode)
                    and (path_metadata.st_dev, path_metadata.st_ino) == stage_identity
                ):
                    os.rmdir(stage_name, dir_fd=docs_fd)
            except OSError:
                pass
        if docs_fd >= 0:
            os.close(docs_fd)
        os.close(root_fd)


def _strict_manifest(manifest_bytes: bytes) -> dict[str, Any]:
    manifest = _json_without_duplicate_keys(manifest_bytes, "manifest_json_invalid")
    _require(
        isinstance(manifest, dict)
        and set(manifest) == {"artifacts", "capture", "schema", "source"}
        and manifest.get("schema") == _MANIFEST_SCHEMA,
        "manifest_structure_invalid",
    )
    _require(_canonical_json(manifest) == manifest_bytes, "manifest_not_canonical")
    artifacts = manifest["artifacts"]
    _require(
        isinstance(artifacts, dict) and set(artifacts) == set(_ARTIFACT_MEDIA_TYPES),
        "manifest_artifacts_invalid",
    )
    for name, expected_media_type in _ARTIFACT_MEDIA_TYPES.items():
        entry = artifacts.get(name)
        _require(
            isinstance(entry, dict)
            and set(entry) == {"bytes", "media_type", "sha256"}
            and isinstance(entry["bytes"], int)
            and 0 < entry["bytes"] <= _MAX_ARTIFACT_BYTES
            and entry["media_type"] == expected_media_type
            and isinstance(entry["sha256"], str)
            and bool(_HEX_SHA256_RE.fullmatch(entry["sha256"])),
            "manifest_artifact_entry_invalid",
        )

    capture = manifest["capture"]
    _require(
        isinstance(capture, dict)
        and set(capture)
        == {
            "execution",
            "network_observation",
            "reader_command",
            "receipt_sha256",
            "runtime",
            "stdout_bytes",
        }
        and capture["reader_command"] == list(_READER_COMMAND)
        and isinstance(capture["stdout_bytes"], int)
        and capture["stdout_bytes"] == artifacts["receipt.json"]["bytes"]
        and capture["receipt_sha256"] == artifacts["receipt.json"]["sha256"],
        "manifest_capture_invalid",
    )
    execution = capture["execution"]
    _require(
        isinstance(execution, dict)
        and set(execution)
        == {
            "bootstrap_sha256",
            "dependency_imports",
            "module",
            "python_flags",
            "snapshot",
        }
        and execution["bootstrap_sha256"] == _sha256(_ISOLATED_BOOTSTRAP.encode("utf-8"))
        and execution["dependency_imports"]
        == "plain interpreter purelib/platlib paths; absolute paths excluded"
        and execution["module"] == _ISOLATED_MODULE
        and execution["python_flags"] == ["-I", "-S", "-B"]
        and execution["snapshot"]
        == "private materialization of committed HEAD; absolute path excluded",
        "manifest_execution_invalid",
    )
    observation = capture["network_observation"]
    _require(
        isinstance(observation, dict)
        and set(observation)
        == {"communication_syscall_count", "scope", "status", "syscalls", "tool"}
        and observation["communication_syscall_count"] == 0
        and observation["status"] == "no_communication_syscalls_observed"
        and observation["syscalls"] == list(_COMMUNICATION_SYSCALLS)
        and observation["scope"] == _NETWORK_OBSERVATION_SCOPE
        and isinstance(observation["tool"], str)
        and bool(_SAFE_RUNTIME_RE.fullmatch(observation["tool"])),
        "manifest_network_observation_invalid",
    )
    runtime = capture["runtime"]
    _require(
        isinstance(runtime, dict)
        and set(runtime)
        == {
            "architecture",
            "dependency_environment_locked",
            "environment_claim",
            "k2do_distribution_version",
            "operating_system",
            "python_implementation",
            "python_version",
        }
        and runtime["dependency_environment_locked"] is True
        and runtime["environment_claim"]
        == "CPython 3.12 CI dependencies are hash-locked by requirements-ci-py312.lock"
        and all(
            isinstance(runtime[key], str) and bool(_SAFE_RUNTIME_RE.fullmatch(runtime[key]))
            for key in (
                "architecture",
                "k2do_distribution_version",
                "operating_system",
                "python_implementation",
                "python_version",
            )
        ),
        "manifest_runtime_invalid",
    )

    source = manifest["source"]
    _require(
        isinstance(source, dict)
        and set(source) == {"commit", "files", "tree", "verification"}
        and isinstance(source["commit"], str)
        and bool(_HEX_OBJECT_RE.fullmatch(source["commit"]))
        and isinstance(source["tree"], str)
        and bool(_HEX_OBJECT_RE.fullmatch(source["tree"]))
        and source["verification"]
        == "current committed blob identities and content hashes; no ancestry requirement"
        and isinstance(source["files"], dict)
        and set(source["files"]) == set(_SOURCE_PATHS),
        "manifest_source_invalid",
    )
    for entry in source["files"].values():
        _require(
            isinstance(entry, dict)
            and set(entry) == {"bytes", "git_blob", "git_mode", "sha256"}
            and isinstance(entry["bytes"], int)
            and 0 < entry["bytes"] <= _MAX_ARTIFACT_BYTES
            and isinstance(entry["git_blob"], str)
            and bool(_HEX_OBJECT_RE.fullmatch(entry["git_blob"]))
            and entry["git_mode"] in {"100644", "100755"}
            and isinstance(entry["sha256"], str)
            and bool(_HEX_SHA256_RE.fullmatch(entry["sha256"])),
            "manifest_source_entry_invalid",
        )
    return manifest


def _verify_capture_object(
    repository: Path,
    source_manifest: Mapping[str, Any],
) -> str:
    commit = source_manifest["commit"]
    tree = source_manifest["tree"]
    commit_available = _git_object_exists(repository, f"{commit}^{{commit}}")
    tree_available = _git_object_exists(repository, f"{tree}^{{tree}}")

    treeish: str | None = None
    status = "unavailable_after_squash"
    if commit_available:
        captured_tree = (
            _run_git(
                repository,
                ["rev-parse", "--verify", f"{commit}^{{tree}}"],
            )
            .decode()
            .strip()
        )
        _require(captured_tree == tree, "capture_commit_tree_mismatch")
        treeish = commit
        status = "verified"
    elif tree_available:
        treeish = tree
        status = "tree_verified_commit_unavailable"

    if treeish is not None:
        for relative_path, expected in source_manifest["files"].items():
            mode, object_id = _tree_entry(repository, treeish, relative_path)
            _require(
                mode == expected["git_mode"] and object_id == expected["git_blob"],
                "capture_object_source_mismatch",
            )
    return status


def _read_published_files(repository: Path) -> dict[str, bytes]:
    root_fd = _open_repository_root(repository)
    docs_fd = -1
    evidence_fd = -1
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            docs_fd = os.open("docs", directory_flags, dir_fd=root_fd)
            evidence_fd = os.open(
                _OUTPUT_DIRECTORY,
                directory_flags,
                dir_fd=docs_fd,
            )
        except OSError as exc:
            raise EvidenceError("evidence_directory_not_safe") from exc
        expected_names = set(_ARTIFACT_MEDIA_TYPES) | {"manifest.json"}
        _require(set(os.listdir(evidence_fd)) == expected_names, "evidence_file_set_changed")
        files: dict[str, bytes] = {}
        for name in sorted(expected_names):
            file_flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                file_flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            try:
                file_fd = os.open(name, file_flags, dir_fd=evidence_fd)
            except OSError as exc:
                raise EvidenceError("evidence_file_not_safe") from exc
            try:
                metadata = os.fstat(file_fd)
                _require(stat.S_ISREG(metadata.st_mode), "evidence_file_not_regular")
                _require(
                    metadata.st_mode & stat.S_IRUSR != 0
                    and metadata.st_mode & 0o111 == 0
                    and metadata.st_mode & 0o022 == 0,
                    "evidence_file_mode_invalid",
                )
                _require(metadata.st_size <= _MAX_ARTIFACT_BYTES, "artifact_exceeded_limit")
                chunks: list[bytes] = []
                remaining = _MAX_ARTIFACT_BYTES + 1
                while remaining > 0:
                    chunk = os.read(file_fd, min(64 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                _require(len(data) <= _MAX_ARTIFACT_BYTES, "artifact_exceeded_limit")
                _require(len(data) == metadata.st_size, "artifact_size_changed_during_read")
                files[name] = data
            finally:
                os.close(file_fd)
        return files
    finally:
        if evidence_fd >= 0:
            os.close(evidence_fd)
        if docs_fd >= 0:
            os.close(docs_fd)
        os.close(root_fd)


def record(
    repository: Path | None = None,
    *,
    capture: CaptureFunction | None = None,
) -> dict[str, Any]:
    """Capture and atomically publish a new absent evidence directory."""

    repository = (repository or _repository_root()).resolve()
    head, tree = _assert_clean_committed_head(repository)
    sources_before = _source_inventory(repository)
    if capture is None:
        with _committed_snapshot(repository, head) as snapshot:
            receipt_bytes, network_observation = _capture_from_snapshot(
                repository,
                snapshot,
            )
    else:
        receipt_bytes, network_observation = capture(repository)
    _validate_captured_receipt(receipt_bytes)
    artifacts = _render_artifacts(receipt_bytes, network_observation)
    sources_after = _source_inventory(repository)
    _require(sources_after == sources_before, "source_changed_during_capture")
    current_head, current_tree = _assert_clean_committed_head(repository)
    _require(
        current_head == head and current_tree == tree,
        "head_changed_during_capture",
    )
    manifest = _build_manifest(
        head=head,
        tree=tree,
        sources=sources_before,
        artifacts=artifacts,
        network_observation=network_observation,
    )
    files = dict(artifacts)
    files["manifest.json"] = _canonical_json(manifest)
    _strict_manifest(files["manifest.json"])

    def verify_before_rename(stage_name: str) -> None:
        _assert_publication_state(
            repository,
            expected_head=head,
            expected_tree=tree,
            expected_sources=sources_before,
            stage_name=stage_name,
        )

    _publish_artifacts(repository, files, pre_publish=verify_before_rename)
    return {
        "artifact_count": len(artifacts),
        "receipt_sha256": _sha256(receipt_bytes),
        "schema": _SCHEMA,
        "source_commit": head,
        "status": "recorded",
    }


def check(
    repository: Path | None = None,
    *,
    capture: CaptureFunction | None = None,
    recapture: bool | None = None,
) -> dict[str, Any]:
    """Verify published artifacts, source blobs, renderers, and optional recapture."""

    repository = (repository or _repository_root()).resolve()
    files = _read_published_files(repository)
    manifest = _strict_manifest(files["manifest.json"])
    capture_object_status = _verify_capture_object(
        repository,
        manifest["source"],
    )
    sources = _source_inventory(repository)
    _require(sources == manifest["source"]["files"], "source_identity_mismatch")

    for name, entry in manifest["artifacts"].items():
        data = files[name]
        _require(
            len(data) == entry["bytes"] and _sha256(data) == entry["sha256"],
            "artifact_identity_mismatch",
        )
    receipt_bytes = files["receipt.json"]
    receipt = _validate_captured_receipt(receipt_bytes)
    observation = manifest["capture"]["network_observation"]
    expected_renderings = _render_artifacts(receipt_bytes, observation)
    for name, expected in expected_renderings.items():
        _require(files[name] == expected, "artifact_rendering_mismatch")

    if recapture is None:
        recapture = platform.system() == "Linux" and os.path.isfile(_STRACE_PATH)
    recapture_status = "not_available"
    if recapture:
        captured_bytes, captured_observation = (capture or _capture_receipt)(repository)
        _require(captured_bytes == receipt_bytes, "fresh_receipt_mismatch")
        _require(
            captured_observation["communication_syscall_count"] == 0
            and captured_observation["status"] == observation["status"]
            and captured_observation["syscalls"] == observation["syscalls"],
            "fresh_network_observation_mismatch",
        )
        recapture_status = "verified"

    _require(receipt["status"] == "verified", "receipt_status_invalid")
    _require(
        _source_inventory(repository) == sources,
        "source_changed_during_check",
    )
    return {
        "artifact_count": len(_ARTIFACT_MEDIA_TYPES),
        "capture_object": capture_object_status,
        "fresh_capture": recapture_status,
        "receipt_sha256": _sha256(receipt_bytes),
        "schema": _SCHEMA,
        "source_blob_count": len(sources),
        "status": "verified",
    }


def _write_public_result(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_canonical_json(dict(value)).decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments not in (["record"], ["check"]):
        _write_public_result(
            sys.stderr,
            {
                "failure": "invalid_invocation",
                "schema": _SCHEMA,
                "status": "failed",
            },
        )
        return 2
    try:
        if arguments == ["record"]:
            result = record()
        else:
            result = check()
    except EvidenceError as exc:
        _write_public_result(
            sys.stderr,
            {
                "failure": exc.code,
                "schema": _SCHEMA,
                "status": "failed",
            },
        )
        return 1
    except Exception:
        _write_public_result(
            sys.stderr,
            {
                "failure": "internal_failure",
                "schema": _SCHEMA,
                "status": "failed",
            },
        )
        return 1
    _write_public_result(sys.stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
