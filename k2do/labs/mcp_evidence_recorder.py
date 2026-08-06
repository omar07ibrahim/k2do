"""Record and verify source-bound visual evidence for the MCP fault laboratory.

The recorder captures the real mcp_fault_lab CLI, renders every visual from
that canonical receipt, and binds the resulting bundle to the exact committed
implementation files that produced it.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw, ImageFont

from k2do.labs import mcp_fault_lab

_SCHEMA: Final = "k2do.mcp-fault-evidence/v1"
_OUTPUT_DIRECTORY: Final = "mcp-fault-evidence"
_COMMAND: Final = ("python", "-m", "k2do.labs.mcp_fault_lab")
_MAX_CAPTURE_BYTES: Final = 512 * 1024
_MAX_ARTIFACT_BYTES: Final = 12 * 1024 * 1024
_SCENARIO_IDS: Final = (
    "happy_discovery_call_restart",
    "catalog_failure_isolated",
    "protocol_error_then_same_connection_recovery",
    "repeated_call_cancellation_recovers",
    "startup_cancellation_reaps_then_restarts",
    "scope_cancellation_drains_borrower_then_restarts",
)
_SOURCE_PATHS: Final = (
    ".github/workflows/ci.yml",
    ".github/workflows/record-mcp-fault-evidence.yml",
    "pyproject.toml",
    "requirements-evidence.txt",
    "k2do/agent/loop.py",
    "k2do/agent/tools/mcp.py",
    "k2do/labs/mcp_evidence_recorder.py",
    "k2do/labs/mcp_fault_lab.py",
    "k2do/labs/strict_mcp_stdio_server.py",
    "tests/test_mcp_evidence_recorder.py",
    "tests/test_mcp_fault_lab.py",
    "tests/test_strict_mcp_stdio_server.py",
)
_MEDIA_TYPES: Final = {
    "architecture.svg": "image/svg+xml",
    "cancellation-timeline.svg": "image/svg+xml",
    "fault-matrix.svg": "image/svg+xml",
    "mcp-fault-lab.txt": "text/plain; charset=utf-8",
    "receipt.json": "application/json",
    "terminal.png": "image/png",
    "terminal.svg": "image/svg+xml",
    "workflow-demo.gif": "image/gif",
}
_SHA_RE: Final = re.compile(r"[0-9a-f]{40,64}\Z")


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be captured or verified safely."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(repository: Path, *arguments: str) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        timeout=20,
    )
    if completed.returncode != 0:
        raise EvidenceError("git_command_failed")
    return completed.stdout


def _head(repository: Path) -> tuple[str, str]:
    head = _git(repository, "rev-parse", "--verify", "HEAD").decode().strip()
    tree = _git(repository, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    _require(bool(_SHA_RE.fullmatch(head)), "head_invalid")
    _require(bool(_SHA_RE.fullmatch(tree)), "tree_invalid")
    return head, tree


def _tree_entry(repository: Path, treeish: str, path: str) -> tuple[str, str]:
    output = _git(repository, "ls-tree", treeish, "--", path).decode()
    fields = output.rstrip("\n").split(None, 3)
    _require(
        len(fields) == 4 and fields[1] == "blob" and fields[3] == path,
        "source_tree_entry_invalid",
    )
    return fields[0], fields[2]


def _source_inventory(
    repository: Path,
    *,
    treeish: str = "HEAD",
    compare_worktree: bool = True,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in _SOURCE_PATHS:
        mode, blob = _tree_entry(repository, treeish, path)
        data = _git(repository, "cat-file", "blob", blob)
        _require(0 < len(data) <= 2 * 1024 * 1024, "source_size_invalid")
        if compare_worktree:
            candidate = repository / path
            _require(
                candidate.is_file() and not candidate.is_symlink(),
                "source_worktree_type_invalid",
            )
            _require(candidate.read_bytes() == data, "source_worktree_content_changed")
        inventory[path] = {
            "bytes": len(data),
            "git_blob": blob,
            "git_mode": mode,
            "sha256": _sha256(data),
        }
    return inventory


def _assert_clean(repository: Path) -> tuple[str, str]:
    _require(
        _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all") == b"",
        "worktree_not_clean",
    )
    return _head(repository)


def _validate_receipt(data: bytes) -> dict[str, Any]:
    _require(
        data.endswith(b"\n") and 0 < len(data) <= _MAX_CAPTURE_BYTES,
        "receipt_encoding_invalid",
    )
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("receipt_json_invalid") from exc
    _require(type(value) is dict, "receipt_root_invalid")
    try:
        rendered = mcp_fault_lab.render_receipt(value).encode()
    except Exception as exc:
        raise EvidenceError("receipt_contract_invalid") from exc
    _require(rendered == data, "receipt_not_canonical")
    _require(
        tuple(item.get("id") for item in value.get("scenarios", ())) == _SCENARIO_IDS,
        "receipt_scenario_order_invalid",
    )
    return value


def _capture(repository: Path) -> tuple[bytes, dict[str, Any]]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "k2do.labs.mcp_fault_lab"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        timeout=90,
    )
    _require(completed.returncode == 0, "capture_process_failed")
    _require(completed.stderr == b"", "capture_stderr_not_empty")
    receipt = _validate_receipt(completed.stdout)
    return completed.stdout, receipt


def _xml(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _transcript(receipt_bytes: bytes) -> bytes:
    return ("$ " + " ".join(_COMMAND) + "\n").encode() + receipt_bytes


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _terminal_lines(receipt_bytes: bytes) -> list[str]:
    return _transcript(receipt_bytes).decode("ascii").splitlines()


def _terminal_svg(receipt_bytes: bytes) -> bytes:
    lines = _terminal_lines(receipt_bytes)
    width = 1520
    line_height = 25
    height = 100 + line_height * len(lines)
    text = []
    for index, line in enumerate(lines):
        color = "#7ee787" if index == 0 else ("#79c0ff" if '"status": "verified"' in line else "#c9d1d9")
        text.append(
            f'<text x="38" y="{76 + index * line_height}" fill="{color}">{_xml(line)}</text>'
        )
    body = "".join(text)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        '<title id="title">Real K2DO MCP fault laboratory output</title>'
        '<desc id="desc">Canonical stdout captured from the committed MCP fault laboratory.</desc>'
        f'<rect width="{width}" height="{height}" rx="18" fill="#0d1117"/>'
        '<circle cx="30" cy="29" r="7" fill="#ff5f56"/><circle cx="54" cy="29" r="7" fill="#ffbd2e"/>'
        '<circle cx="78" cy="29" r="7" fill="#27c93f"/>'
        '<text x="760" y="35" text-anchor="middle" fill="#8b949e" '
        'font-family="DejaVu Sans Mono,monospace" font-size="15">captured stdout · no synthetic values</text>'
        f'<g font-family="DejaVu Sans Mono,monospace" font-size="15">{body}</g></svg>\n'
    ).encode()


def _terminal_png(receipt_bytes: bytes) -> bytes:
    lines = _terminal_lines(receipt_bytes)
    line_height = 28
    image = Image.new("RGB", (1600, 110 + line_height * len(lines)), "#0d1117")
    draw = ImageDraw.Draw(image)
    font = _font(18)
    draw.ellipse((24, 23, 40, 39), fill="#ff5f56")
    draw.ellipse((50, 23, 66, 39), fill="#ffbd2e")
    draw.ellipse((76, 23, 92, 39), fill="#27c93f")
    draw.text((800, 24), "captured stdout · source-bound", fill="#8b949e", font=font, anchor="ma")
    for index, line in enumerate(lines):
        color = "#7ee787" if index == 0 else ("#79c0ff" if '"status": "verified"' in line else "#c9d1d9")
        draw.text((30, 72 + index * line_height), line, fill=color, font=font)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _architecture_svg(receipt: Mapping[str, Any]) -> bytes:
    scenarios = receipt["scenarios"]
    generations = sum(item["generations"] for item in scenarios)
    reaped = sum(item["cleanup"]["processes_reaped"] for item in scenarios)
    nodes = (
        (70, "Fault scenarios", f"{len(scenarios)} deterministic cases"),
        (410, "AgentLoop lifecycle", "production mcp_lifespan / close_mcp"),
        (750, "Pinned MCP SDK", receipt["protocol"]["client"]),
        (1090, "Strict stdio fixtures", f"{generations} fresh generations"),
    )
    boxes = []
    arrows = []
    for index, (x, title, subtitle) in enumerate(nodes):
        boxes.append(
            f'<rect x="{x}" y="220" width="280" height="150" rx="20" fill="#161b22" stroke="#58a6ff" stroke-width="2"/>'
            f'<text x="{x + 140}" y="270" text-anchor="middle" fill="#f0f6fc" font-size="24" font-weight="700">{_xml(title)}</text>'
            f'<text x="{x + 140}" y="310" text-anchor="middle" fill="#8b949e" font-size="15">{_xml(subtitle)}</text>'
        )
        if index:
            arrows.append(
                f'<path d="M {x - 60} 295 H {x - 10}" stroke="#7ee787" stroke-width="4" marker-end="url(#arrow)"/>'
            )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="650" viewBox="0 0 1440 650" '
        'role="img" aria-labelledby="title desc"><title id="title">MCP fault laboratory architecture</title>'
        f'<desc id="desc">Architecture derived from a verified receipt with {len(scenarios)} scenarios and {reaped} reaped subprocesses.</desc>'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#7ee787"/></marker></defs>'
        '<rect width="1440" height="650" rx="24" fill="#0d1117"/>'
        '<text x="70" y="78" fill="#f0f6fc" font-size="38" font-weight="700">Real subprocesses. Production lifecycle. Safe public receipt.</text>'
        f'<text x="70" y="125" fill="#8b949e" font-size="20">{generations} process generations · {reaped} reaped · stdio NDJSON · private authenticated AF_UNIX audit</text>'
        + "".join(arrows)
        + "".join(boxes)
        + '<rect x="310" y="470" width="820" height="90" rx="18" fill="#12261e" stroke="#238636"/>'
        + '<text x="720" y="510" text-anchor="middle" fill="#7ee787" font-size="22" font-weight="700">Public boundary: labels, counts, booleans, SHA-256 only</text>'
        + '<text x="720" y="540" text-anchor="middle" fill="#8b949e" font-size="16">No token, PID, request ID, raw error, path, argument, or timing leaves the lab</text></svg>\n'
    ).encode()


def _fault_matrix_svg(receipt: Mapping[str, Any]) -> bytes:
    scenarios = receipt["scenarios"]
    height = 185 + len(scenarios) * 82
    rows = []
    for index, scenario in enumerate(scenarios):
        y = 155 + index * 82
        fill = "#161b22" if index % 2 == 0 else "#111820"
        cleanup = scenario["cleanup"]
        rows.append(
            f'<rect x="40" y="{y}" width="1420" height="70" rx="10" fill="{fill}"/>'
            f'<text x="65" y="{y + 29}" fill="#f0f6fc" font-size="18" font-weight="700">{_xml(scenario["id"])}</text>'
            f'<text x="65" y="{y + 53}" fill="#8b949e" font-size="14">{_xml(scenario["recovery"])}</text>'
            f'<text x="875" y="{y + 42}" fill="#79c0ff" font-size="18">{scenario["generations"]}</text>'
            f'<text x="1045" y="{y + 42}" fill="#7ee787" font-size="18">{cleanup["processes_reaped"]}/{scenario["generations"]}</text>'
            f'<text x="1255" y="{y + 42}" fill="#7ee787" font-size="18">restored</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{height}" viewBox="0 0 1500 {height}" '
        'role="img" aria-labelledby="title desc"><title id="title">Verified MCP fault matrix</title>'
        '<desc id="desc">Each row is derived from a real scenario receipt.</desc>'
        f'<rect width="1500" height="{height}" rx="24" fill="#0d1117"/>'
        '<text x="40" y="58" fill="#f0f6fc" font-size="36" font-weight="700">Failure and recovery matrix</text>'
        '<text x="65" y="128" fill="#8b949e" font-size="16">scenario / observed recovery</text>'
        '<text x="850" y="128" fill="#8b949e" font-size="16">generations</text>'
        '<text x="1010" y="128" fill="#8b949e" font-size="16">reaped</text>'
        '<text x="1220" y="128" fill="#8b949e" font-size="16">registry</text>'
        + "".join(rows)
        + "</svg>\n"
    ).encode()


def _timeline_svg(receipt: Mapping[str, Any]) -> bytes:
    scenarios = {item["id"]: item for item in receipt["scenarios"]}
    stages = (
        ("discover", "Modern server/discover"),
        ("isolate", scenarios["catalog_failure_isolated"]["recovery"]),
        ("recover", scenarios["protocol_error_then_same_connection_recovery"]["recovery"]),
        ("cancel ×2", f'{scenarios["repeated_call_cancellation_recovers"]["courtesy_cancellations"]} courtesy notifications'),
        ("restart", scenarios["startup_cancellation_reaps_then_restarts"]["recovery"]),
        ("drain", scenarios["scope_cancellation_drains_borrower_then_restarts"]["cancellation"]),
    )
    items = []
    for index, (label, detail) in enumerate(stages):
        x = 120 + index * 255
        items.append(
            f'<circle cx="{x}" cy="285" r="25" fill="#238636" stroke="#7ee787" stroke-width="3"/>'
            f'<text x="{x}" y="235" text-anchor="middle" fill="#f0f6fc" font-size="20" font-weight="700">{_xml(label)}</text>'
            f'<text x="{x}" y="345" text-anchor="middle" fill="#8b949e" font-size="13">{_xml(detail)}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="560" viewBox="0 0 1600 560" '
        'role="img" aria-labelledby="title desc"><title id="title">MCP cancellation and recovery timeline</title>'
        '<desc id="desc">Observed sequence summarized from six real lifecycle scenarios.</desc>'
        '<rect width="1600" height="560" rx="24" fill="#0d1117"/>'
        '<text x="55" y="68" fill="#f0f6fc" font-size="36" font-weight="700">Cancellation is a lifecycle, not an exception branch</text>'
        '<text x="55" y="110" fill="#8b949e" font-size="18">Every transition below is asserted against a real stdio subprocess.</text>'
        '<path d="M 120 285 H 1395" stroke="#58a6ff" stroke-width="6"/>'
        + "".join(items)
        + '<rect x="380" y="425" width="840" height="70" rx="16" fill="#12261e" stroke="#238636"/>'
        + '<text x="800" y="468" text-anchor="middle" fill="#7ee787" font-size="21">EOF observed · process reaped · tool registry restored · same loop reusable</text>'
        + "</svg>\n"
    ).encode()


def _workflow_gif(receipt: Mapping[str, Any]) -> bytes:
    scenarios = receipt["scenarios"]
    frames: list[Image.Image] = []
    font = _font(22)
    title_font = _font(34)
    for visible in range(1, len(scenarios) + 1):
        image = Image.new("RGB", (1280, 760), "#0d1117")
        draw = ImageDraw.Draw(image)
        draw.text((54, 45), "K2DO MCP fault laboratory", fill="#f0f6fc", font=title_font)
        draw.text(
            (54, 96),
            f"verified scenario {visible}/{len(scenarios)} · real subprocess receipt",
            fill="#8b949e",
            font=font,
        )
        for index, scenario in enumerate(scenarios):
            y = 155 + index * 82
            active = index < visible
            draw.rounded_rectangle(
                (54, y, 1225, y + 62),
                radius=12,
                fill="#12261e" if active else "#161b22",
                outline="#238636" if active else "#30363d",
                width=2,
            )
            draw.text(
                (78, y + 18),
                scenario["id"],
                fill="#7ee787" if active else "#6e7681",
                font=font,
            )
            state = scenario["recovery"] if active else "pending"
            draw.text((930, y + 18), state, fill="#79c0ff" if active else "#6e7681", font=font)
        draw.text(
            (54, 690),
            "Source-bound output · categorical failures · deterministic public projection",
            fill="#c9d1d9",
            font=font,
        )
        frames.append(image)
    output = io.BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=700,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output.getvalue()


def _validate_artifact(name: str, data: bytes) -> None:
    _require(0 < len(data) <= _MAX_ARTIFACT_BYTES, "artifact_size_invalid")
    if name.endswith(".svg"):
        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            raise EvidenceError("artifact_svg_invalid") from exc
        _require(root.tag.endswith("svg"), "artifact_svg_root_invalid")
    elif name.endswith((".png", ".gif")):
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                _require(image.width >= 1000 and image.height >= 500, "artifact_raster_too_small")
                if name.endswith(".gif"):
                    _require(getattr(image, "n_frames", 1) >= 6, "artifact_gif_frames_invalid")
        except (OSError, SyntaxError) as exc:
            raise EvidenceError("artifact_raster_invalid") from exc


def _render_artifacts(receipt_bytes: bytes, receipt: Mapping[str, Any]) -> dict[str, bytes]:
    artifacts = {
        "architecture.svg": _architecture_svg(receipt),
        "cancellation-timeline.svg": _timeline_svg(receipt),
        "fault-matrix.svg": _fault_matrix_svg(receipt),
        "mcp-fault-lab.txt": _transcript(receipt_bytes),
        "receipt.json": receipt_bytes,
        "terminal.png": _terminal_png(receipt_bytes),
        "terminal.svg": _terminal_svg(receipt_bytes),
        "workflow-demo.gif": _workflow_gif(receipt),
    }
    _require(set(artifacts) == set(_MEDIA_TYPES), "artifact_set_invalid")
    for name, data in artifacts.items():
        _validate_artifact(name, data)
    return artifacts


def _manifest(
    *,
    head: str,
    tree: str,
    sources: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, bytes],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "status": "verified",
        "source": {
            "commit": head,
            "tree": tree,
            "files": sources,
            "verification": "current committed blobs must remain byte-identical",
        },
        "capture": {
            "command": list(_COMMAND),
            "stdout": "canonical receipt.json",
            "stderr_bytes": 0,
            "fixture_transport": receipt["evidence_boundary"]["fixture_io"],
            "network_observation": (
                "No independent syscall claim: the lab uses stdio plus a private authenticated "
                "AF_UNIX lifecycle oracle and exposes no network endpoint."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "mcp": importlib.metadata.version("mcp"),
            "pillow": importlib.metadata.version("Pillow"),
        },
        "artifacts": {
            name: {
                "bytes": len(data),
                "media_type": _MEDIA_TYPES[name],
                "sha256": _sha256(data),
            }
            for name, data in sorted(artifacts.items())
        },
    }


def _publish(repository: Path, files: Mapping[str, bytes], *, replace: bool) -> None:
    docs = repository / "docs"
    destination = docs / _OUTPUT_DIRECTORY
    _require(docs.is_dir() and not docs.is_symlink(), "docs_directory_invalid")
    if destination.exists():
        _require(replace and destination.is_dir() and not destination.is_symlink(), "destination_exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{_OUTPUT_DIRECTORY}-", dir=docs))
    backup: Path | None = None
    try:
        for name, data in sorted(files.items()):
            _require("/" not in name, "artifact_name_invalid")
            path = stage / name
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        if destination.exists():
            backup = docs / f".{_OUTPUT_DIRECTORY}-old-{os.getpid()}"
            _require(not backup.exists(), "backup_exists")
            destination.rename(backup)
        stage.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if not destination.exists() and backup is not None and backup.exists():
            backup.rename(destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def record(repository: Path | None = None, *, replace: bool = False) -> dict[str, Any]:
    repository = (repository or _repository_root()).resolve()
    head, tree = _assert_clean(repository)
    sources = _source_inventory(repository)
    receipt_bytes, receipt = _capture(repository)
    artifacts = _render_artifacts(receipt_bytes, receipt)
    current_head, current_tree = _head(repository)
    _require((current_head, current_tree) == (head, tree), "head_changed_during_capture")
    _require(_source_inventory(repository) == sources, "source_changed_during_capture")
    manifest = _manifest(
        head=head,
        tree=tree,
        sources=sources,
        artifacts=artifacts,
        receipt=receipt,
    )
    files = dict(artifacts)
    files["manifest.json"] = _canonical_json(manifest)
    _publish(repository, files, replace=replace)
    return {"commit": head, "files": len(files), "status": "recorded"}


def _read_bundle(repository: Path) -> dict[str, bytes]:
    directory = repository / "docs" / _OUTPUT_DIRECTORY
    _require(directory.is_dir() and not directory.is_symlink(), "evidence_directory_invalid")
    expected = set(_MEDIA_TYPES) | {"manifest.json"}
    _require({path.name for path in directory.iterdir()} == expected, "evidence_file_set_changed")
    files: dict[str, bytes] = {}
    for name in sorted(expected):
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), "evidence_file_type_invalid")
        data = path.read_bytes()
        _require(0 < len(data) <= _MAX_ARTIFACT_BYTES, "evidence_file_size_invalid")
        files[name] = data
    return files


def check(repository: Path | None = None, *, fresh: bool = False) -> dict[str, Any]:
    repository = (repository or _repository_root()).resolve()
    files = _read_bundle(repository)
    try:
        manifest = json.loads(files["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("manifest_json_invalid") from exc
    _require(type(manifest) is dict and manifest.get("schema") == _SCHEMA, "manifest_schema_invalid")
    _require(manifest.get("status") == "verified", "manifest_status_invalid")
    sources = _source_inventory(repository)
    _require(manifest.get("source", {}).get("files") == sources, "manifest_source_changed")
    expected_artifacts = manifest.get("artifacts")
    _require(type(expected_artifacts) is dict and set(expected_artifacts) == set(_MEDIA_TYPES), "manifest_artifacts_invalid")
    for name in _MEDIA_TYPES:
        metadata = expected_artifacts[name]
        _require(
            metadata
            == {
                "bytes": len(files[name]),
                "media_type": _MEDIA_TYPES[name],
                "sha256": _sha256(files[name]),
            },
            "manifest_artifact_changed",
        )
        _validate_artifact(name, files[name])
    receipt = _validate_receipt(files["receipt.json"])
    rendered = _render_artifacts(files["receipt.json"], receipt)
    _require(all(files[name] == data for name, data in rendered.items()), "artifact_render_changed")
    if fresh:
        fresh_bytes, _fresh_receipt = _capture(repository)
        _require(fresh_bytes == files["receipt.json"], "fresh_receipt_changed")
    return {
        "commit": manifest["source"]["commit"],
        "files": len(files),
        "fresh_capture": fresh,
        "status": "verified",
    }


def _write_result(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_canonical_json(value).decode())


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["record"]:
            result = record()
        elif args == ["record", "--replace"]:
            result = record(replace=True)
        elif args == ["check"]:
            result = check()
        elif args == ["check", "--fresh"]:
            result = check(fresh=True)
        else:
            _write_result(
                sys.stderr,
                {"failure": "invalid_invocation", "schema": _SCHEMA, "status": "failed"},
            )
            return 2
    except Exception:
        _write_result(
            sys.stderr,
            {"failure": "verification_failed", "schema": _SCHEMA, "status": "failed"},
        )
        return 1
    _write_result(sys.stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
