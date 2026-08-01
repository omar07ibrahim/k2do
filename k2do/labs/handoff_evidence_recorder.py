"""Capture and verify source-bound evidence for the routed handoff lab.

``record`` requires a clean committed tree and an absent destination.  It
executes the committed package from a private snapshot, rejects communication
syscalls with ``strace``, renders every visual from the captured receipt, and
publishes the bundle atomically.  ``check`` verifies source blobs, artifact
hashes, renderer output, image structure, and a fresh capture when available.
"""

from __future__ import annotations

import html
import importlib.metadata
import io
import json
import os
import platform
import re
import secrets
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw, ImageFont, features

from k2do.labs import agent_handoff_trace
from k2do.labs import trace_evidence_recorder as evidence_core

_SCHEMA: Final = "k2do.handoff-evidence/v1"
_MANIFEST_SCHEMA: Final = "k2do.handoff-evidence-manifest/v1"
_OUTPUT_DIRECTORY: Final = "agent-handoff-evidence"
_MAX_SOURCE_BYTES: Final = 2 * 1024 * 1024
_MAX_ARTIFACT_BYTES: Final = 12 * 1024 * 1024
_COMMAND_TIMEOUT_S: Final = 20.0
_ISOLATED_MODULE: Final = "k2do.labs.agent_handoff_trace"
_ISOLATED_BOOTSTRAP: Final = (
    "import runpy,sys;"
    "snapshot=sys.argv[1];"
    "dependencies=sys.argv[2:];"
    "sys.path[:0]=[snapshot,*dependencies];"
    f"sys.argv=['{_ISOLATED_MODULE}'];"
    f"runpy.run_module('{_ISOLATED_MODULE}',run_name='__main__')"
)
_READER_COMMAND: Final = ("python", "-m", _ISOLATED_MODULE)
_TRANSCRIPT_COMMAND: Final = "$ python -m k2do.labs.agent_handoff_trace\n"
_NETWORK_OBSERVATION_SCOPE: Final = (
    "one Linux strace run of the committed handoff lab and all child threads; "
    "the listed communication syscalls exclude passive event-loop bookkeeping"
)
_ARTIFACT_MEDIA_TYPES: Final = {
    "architecture.svg": "image/svg+xml",
    "contract-matrix.svg": "image/svg+xml",
    "handoff-lab.txt": "text/plain; charset=utf-8",
    "receipt.json": "application/json",
    "terminal.png": "image/png",
    "terminal.svg": "image/svg+xml",
    "tool-timeline.svg": "image/svg+xml",
    "workflow-demo.gif": "image/gif",
}
_HEX_OBJECT_RE: Final = re.compile(r"[0-9a-f]{40,64}")
_HEX_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_SAFE_RUNTIME_RE: Final = re.compile(r"[A-Za-z0-9_.+() -]{1,120}")

EvidenceError = evidence_core.EvidenceError
CaptureFunction = Callable[[Path], tuple[bytes, dict[str, Any]]]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _sha256(data: bytes) -> str:
    return evidence_core._sha256(data)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_paths(repository: Path, head: str) -> tuple[str, ...]:
    package_paths = [
        path
        for path, _mode, _object_id in evidence_core._committed_package_entries(repository, head)
    ]
    paths = ("pyproject.toml", *package_paths)
    _require(len(paths) == len(set(paths)), "source_path_duplicate")
    _require(list(paths[1:]) == sorted(paths[1:]), "source_paths_not_canonical")
    return paths


def _source_inventory(
    repository: Path,
    source_paths: Sequence[str],
    *,
    treeish: str = "HEAD",
    compare_worktree: bool = True,
) -> dict[str, dict[str, Any]]:
    root_fd = evidence_core._open_repository_root(repository)
    try:
        inventory: dict[str, dict[str, Any]] = {}
        for relative_path in source_paths:
            mode, object_id = evidence_core._tree_entry(repository, treeish, relative_path)
            committed = evidence_core._run_git(repository, ["cat-file", "blob", object_id])
            _require(0 < len(committed) <= _MAX_SOURCE_BYTES, "source_size_invalid")
            if compare_worktree:
                worktree = evidence_core._read_relative_regular(
                    root_fd,
                    relative_path,
                    maximum_bytes=_MAX_SOURCE_BYTES,
                )
                _require(worktree == committed, "source_worktree_content_mismatch")
            inventory[relative_path] = {
                "bytes": len(committed),
                "git_blob": object_id,
                "git_mode": mode,
                "sha256": _sha256(committed),
            }
        return inventory
    finally:
        os.close(root_fd)


def _json_without_duplicates(data: bytes, code: str) -> Any:
    return evidence_core._json_without_duplicate_keys(data, code)


def _validate_receipt(receipt_bytes: bytes) -> dict[str, Any]:
    _require(
        receipt_bytes.endswith(b"\n") and len(receipt_bytes) <= 256 * 1024,
        "receipt_encoding_invalid",
    )
    receipt = _json_without_duplicates(receipt_bytes, "receipt_json_invalid")
    _require(isinstance(receipt, dict), "receipt_root_invalid")
    try:
        rendered = agent_handoff_trace.render_receipt(receipt).encode()
    except Exception as exc:
        raise EvidenceError("receipt_contract_invalid") from exc
    _require(rendered == receipt_bytes, "receipt_not_canonical")
    return receipt


def _capture_from_snapshot(
    repository: Path,
    snapshot: Path,
) -> tuple[bytes, dict[str, Any]]:
    _require(platform.system() == "Linux", "strace_capture_requires_linux")
    strace_path = evidence_core._validate_strace_binary()
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _ISOLATED_BOOTSTRAP,
        str(snapshot),
        *evidence_core._dependency_import_paths(),
    ]
    environment = evidence_core._capture_environment()
    direct_code, direct_stdout, direct_stderr = evidence_core._run_bounded(
        command,
        cwd=snapshot,
        environment=environment,
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    _require(direct_code == 0 and direct_stderr == b"", "handoff_lab_capture_failed")
    _validate_receipt(direct_stdout)

    syscalls = list(evidence_core._COMMUNICATION_SYSCALLS)
    traced_code, traced_stdout, traced_stderr = evidence_core._run_bounded(
        [
            strace_path,
            "-f",
            "-qq",
            "-e",
            f"trace={','.join(syscalls)}",
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
    return direct_stdout, {
        "communication_syscall_count": 0,
        "scope": _NETWORK_OBSERVATION_SCOPE,
        "status": "no_communication_syscalls_observed",
        "syscalls": syscalls,
        "tool": evidence_core._strace_version(strace_path, repository),
    }


def _capture_receipt(repository: Path) -> tuple[bytes, dict[str, Any]]:
    head = evidence_core._run_git(repository, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    _require(bool(_HEX_OBJECT_RE.fullmatch(head)), "head_identity_invalid")
    with evidence_core._committed_snapshot(repository, head) as snapshot:
        return _capture_from_snapshot(repository, snapshot)


def _xml_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_transcript(receipt_bytes: bytes) -> bytes:
    return _TRANSCRIPT_COMMAND.encode() + receipt_bytes


def _transcript_lines(receipt_bytes: bytes) -> list[str]:
    transcript = _render_transcript(receipt_bytes).decode("ascii")
    lines = transcript.splitlines()
    _require(lines and lines[0] == _TRANSCRIPT_COMMAND.rstrip(), "transcript_command_changed")
    _require("\n".join(lines[1:]) + "\n" == receipt_bytes.decode("ascii"), "transcript_changed")
    return lines


def _line_color(line: str) -> str:
    if line.startswith("$ "):
        return "#86efac"
    if "sha256" in line or _HEX_SHA256_RE.search(line):
        return "#fbbf24"
    if '"verified"' in line or '"persisted": true' in line:
        return "#7dd3fc"
    return "#dbeafe"


def _render_terminal_svg(receipt_bytes: bytes) -> bytes:
    lines = _transcript_lines(receipt_bytes)
    line_height = 22
    top = 112
    height = top + len(lines) * line_height + 48
    text = "\n".join(
        f'<text x="62" y="{top + index * line_height}" class="mono" '
        f'font-size="15" fill="{_line_color(line)}">{_xml_text(line)}</text>'
        for index, line in enumerate(lines)
    )
    body = f"""  <g filter="url(#shadow)">
    <rect x="28" y="28" width="1484" height="{height - 56}" rx="18" fill="#020617" stroke="#334155"/>
    <rect x="28" y="28" width="1484" height="54" rx="18" fill="#111c2f"/>
    <rect x="28" y="64" width="1484" height="18" fill="#111c2f"/>
    <circle cx="62" cy="55" r="7" fill="#fb7185"/>
    <circle cx="86" cy="55" r="7" fill="#fbbf24"/>
    <circle cx="110" cy="55" r="7" fill="#4ade80"/>
    <text x="770" y="62" text-anchor="middle" class="sans" font-size="16" fill="#94a3b8">exact captured stdout · canonical offline receipt</text>
{text}
  </g>"""
    return evidence_core._svg_document(
        width=1540,
        height=height,
        title="K2DO routed handoff CLI capture",
        description=(
            "A line-for-line rendering of the captured agent_handoff_trace stdout. "
            "The matching text artifact preserves the exact bytes."
        ),
        body=body,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _render_terminal_png(receipt_bytes: bytes) -> bytes:
    lines = _transcript_lines(receipt_bytes)
    font = _load_font(16)
    line_height = 22
    longest = max(float(font.getlength(line)) for line in lines)
    width = max(1200, min(2200, int(longest) + 96))
    height = 108 + len(lines) * line_height + 42
    image = Image.new("RGB", (width, height), "#07111f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, width - 20, height - 20), 18, fill="#020617", outline="#334155")
    draw.rounded_rectangle((20, 20, width - 20, 78), 18, fill="#111c2f")
    draw.rectangle((20, 58, width - 20, 78), fill="#111c2f")
    for x, color in ((54, "#fb7185"), (78, "#fbbf24"), (102, "#4ade80")):
        draw.ellipse((x - 7, 42, x + 7, 56), fill=color)
    title_font = _load_font(17)
    title = "exact captured stdout | canonical offline receipt"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((width - title_width) / 2, 39), title, font=title_font, fill="#94a3b8")
    for index, line in enumerate(lines):
        draw.text((48, 96 + index * line_height), line, font=font, fill=_line_color(line))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _render_architecture_svg(receipt: Mapping[str, Any]) -> bytes:
    nodes = receipt["workflow"]["nodes"]
    _require(isinstance(nodes, list) and len(nodes) == 9, "workflow_nodes_changed")
    positions = [
        (54, 110),
        (394, 110),
        (734, 110),
        (1074, 110),
        (1074, 340),
        (734, 340),
        (394, 340),
        (54, 340),
        (54, 570),
    ]
    arrows: list[str] = []
    for index in range(len(positions) - 1):
        x1, y1 = positions[index]
        x2, y2 = positions[index + 1]
        if y1 == y2 and x2 > x1:
            start, end = (x1 + 260, y1 + 70), (x2 - 18, y2 + 70)
        elif y1 == y2:
            start, end = (x1 - 18, y1 + 70), (x2 + 260, y2 + 70)
        else:
            start, end = (x1 + 130, y1 + 140), (x2 + 130, y2 - 18)
        arrows.append(
            f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" '
            'stroke="#7dd3fc" stroke-width="4" marker-end="url(#arrow)"/>'
        )
    cards = []
    for index, (node, (x, y)) in enumerate(zip(nodes, positions, strict=True), 1):
        group = node.split(".", 1)[0]
        detail = node.split(".", 1)[1]
        accent = "#22d3ee" if group in {"thinkers", "judge"} else "#a78bfa"
        cards.append(
            f'''<g filter="url(#shadow)">
  <rect x="{x}" y="{y}" width="260" height="140" rx="18" fill="#0f1d33" stroke="{accent}" stroke-width="2"/>
  <text x="{x + 22}" y="{y + 34}" class="mono" font-size="14" fill="#94a3b8">STEP {index:02d}</text>
  <text x="{x + 22}" y="{y + 72}" class="sans" font-size="21" font-weight="700" fill="#f8fafc">{_xml_text(group)}</text>
  <text x="{x + 22}" y="{y + 104}" class="mono" font-size="16" fill="#7dd3fc">{_xml_text(detail)}</text>
</g>'''
        )
    body = f"""  <text x="54" y="60" class="sans" font-size="30" font-weight="700" fill="#f8fafc">Routed DeepThink handoff — observed production path</text>
  <text x="54" y="88" class="sans" font-size="16" fill="#94a3b8">Every node and edge is read from the canonical receipt; arrows show execution order.</text>
  {"".join(arrows)}
  {"".join(cards)}
  <text x="394" y="690" class="mono" font-size="16" fill="#86efac">route=deepthink · parallel peak=3 · tools=write_file → read_file · session=persisted</text>"""
    return evidence_core._svg_document(
        width=1400,
        height=750,
        title="K2DO routed handoff architecture",
        description="Receipt-derived architecture from inbound message through persisted session and outbound response.",
        body=body,
    )


def _render_timeline_svg(receipt: Mapping[str, Any]) -> bytes:
    deepthink = receipt["deepthink"]
    handoff = receipt["handoff"]
    session = receipt["session"]
    stages = [
        ("ROUTE", "deepthink", "production classifier at threshold 0.6", "#a78bfa"),
        (
            "FAN-OUT",
            f"{deepthink['thinker_count']} thinkers",
            "strict barrier; peak provider calls = 3",
            "#22d3ee",
        ),
        ("JUDGE", deepthink["selected"], deepthink["judge_gate"].replace("_", " "), "#38bdf8"),
        ("HANDOFF", "3 provider turns", "full tool catalog validated each turn", "#60a5fa"),
        (
            "WRITE",
            handoff["tool_sequence"][0],
            f"{handoff['artifact']['bytes']} byte bounded artifact",
            "#4ade80",
        ),
        (
            "READ-BACK",
            handoff["tool_sequence"][1],
            "exact tool-result payload validated",
            "#34d399",
        ),
        (
            "PERSIST",
            " + ".join(session["tools_used"]),
            "fresh SessionManager disk reload",
            "#fbbf24",
        ),
    ]
    cards: list[str] = []
    for index, (label, value, detail, color) in enumerate(stages):
        y = 104 + index * 112
        cards.append(
            f'''<circle cx="90" cy="{y + 38}" r="18" fill="{color}"/>
<text x="90" y="{y + 44}" text-anchor="middle" class="mono" font-size="14" font-weight="700" fill="#020617">{index + 1}</text>
<rect x="138" y="{y}" width="1110" height="82" rx="16" fill="#0f1d33" stroke="#334155"/>
<text x="166" y="{y + 28}" class="mono" font-size="14" fill="{color}">{_xml_text(label)}</text>
<text x="166" y="{y + 56}" class="sans" font-size="20" font-weight="700" fill="#f8fafc">{_xml_text(value)}</text>
<text x="700" y="{y + 51}" class="sans" font-size="17" fill="#cbd5e1">{_xml_text(detail)}</text>'''
        )
    body = f'''  <text x="60" y="54" class="sans" font-size="30" font-weight="700" fill="#f8fafc">Execution timeline · reasoning becomes a verified side effect</text>
  <line x1="90" y1="142" x2="90" y2="{104 + (len(stages) - 1) * 112 + 38}" stroke="#334155" stroke-width="5"/>
  {"".join(cards)}'''
    return evidence_core._svg_document(
        width=1320,
        height=104 + len(stages) * 112,
        title="K2DO tool handoff timeline",
        description="A receipt-derived timeline of routing, parallel reasoning, tool execution, read-back, and persistence.",
        body=body,
    )


def _render_contract_matrix_svg(
    receipt: Mapping[str, Any],
    network_observation: Mapping[str, Any],
) -> bytes:
    cards = [
        ("ROUTER", "deepthink", "default threshold 0.6"),
        ("CONCURRENCY", "3 active", "barrier-observed peak"),
        ("JUDGE GATE", "terminal first", "all thinkers drained"),
        ("TOOL CHAIN", "write → read", "workspace restricted"),
        ("SESSION", "persisted", "disk reload verified"),
        (
            "NETWORK",
            f"{network_observation['communication_syscall_count']} observed",
            "listed strace set",
        ),
    ]
    markup: list[str] = []
    for index, (label, value, detail) in enumerate(cards):
        column, row = index % 3, index // 3
        x, y = 54 + column * 420, 120 + row * 230
        markup.append(
            f'''<g filter="url(#shadow)">
  <rect x="{x}" y="{y}" width="370" height="180" rx="20" fill="#0f1d33" stroke="#334155"/>
  <text x="{x + 26}" y="{y + 38}" class="mono" font-size="14" fill="#94a3b8">{_xml_text(label)}</text>
  <text x="{x + 26}" y="{y + 92}" class="sans" font-size="29" font-weight="700" fill="#7dd3fc">{_xml_text(value)}</text>
  <text x="{x + 26}" y="{y + 132}" class="sans" font-size="16" fill="#cbd5e1">{_xml_text(detail)}</text>
</g>'''
        )
    receipt_digest = _sha256(agent_handoff_trace.render_receipt(dict(receipt)).encode())
    body = f"""  <text x="54" y="56" class="sans" font-size="30" font-weight="700" fill="#f8fafc">Verified contract matrix</text>
  <text x="54" y="86" class="sans" font-size="16" fill="#94a3b8">Claims are deliberately narrower than a live model benchmark or MCP lifecycle test.</text>
  {"".join(markup)}
  <text x="54" y="620" class="mono" font-size="15" fill="#fbbf24">receipt sha256  {receipt_digest}</text>"""
    return evidence_core._svg_document(
        width=1320,
        height=670,
        title="K2DO routed handoff contract matrix",
        description="Six evidence-backed properties derived from the handoff receipt and strace observation.",
        body=body,
    )


def _draw_demo_frame(
    receipt: Mapping[str, Any],
    active_index: int,
) -> Image.Image:
    nodes = receipt["workflow"]["nodes"]
    image = Image.new("RGB", (1200, 675), "#08111f")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(30)
    label_font = _load_font(17)
    small_font = _load_font(14)
    draw.text(
        (48, 40), "K2DO routed handoff | verified workflow replay", font=title_font, fill="#f8fafc"
    )
    draw.text(
        (50, 82),
        "Frames reveal receipt nodes in observed execution order; timing is illustrative.",
        font=small_font,
        fill="#94a3b8",
    )
    positions = [
        (60, 140),
        (440, 140),
        (820, 140),
        (820, 295),
        (440, 295),
        (60, 295),
        (60, 450),
        (440, 450),
        (820, 450),
    ]
    for index in range(len(nodes) - 1):
        x1, y1 = positions[index]
        x2, y2 = positions[index + 1]
        if y1 == y2 and x2 > x1:
            points = [(x1 + 300, y1 + 50), (x2 - 20, y2 + 50)]
        elif y1 == y2:
            points = [(x1 - 20, y1 + 50), (x2 + 300, y2 + 50)]
        else:
            points = [(x1 + 150, y1 + 100), (x2 + 150, y2 - 20)]
        draw.line(points, fill="#334155", width=5)
    for index, (node, (x, y)) in enumerate(zip(nodes, positions, strict=True)):
        active = index <= active_index
        fill = "#123452" if active else "#0b1728"
        outline = "#22d3ee" if active else "#334155"
        draw.rounded_rectangle((x, y, x + 300, y + 100), 18, fill=fill, outline=outline, width=3)
        group, detail = node.split(".", 1)
        draw.text((x + 20, y + 20), group, font=label_font, fill="#f8fafc" if active else "#64748b")
        draw.text(
            (x + 20, y + 58), detail, font=small_font, fill="#7dd3fc" if active else "#475569"
        )
        draw.text((x + 260, y + 18), f"{index + 1:02d}", font=small_font, fill=outline)
    current = nodes[active_index].replace(".", " / ")
    draw.rounded_rectangle((48, 610, 1152, 648), 12, fill="#020617", outline="#334155")
    draw.text(
        (68, 620), f"observed step {active_index + 1}/9  {current}", font=label_font, fill="#86efac"
    )
    return image


def _render_workflow_gif(receipt: Mapping[str, Any]) -> bytes:
    frames = [_draw_demo_frame(receipt, index) for index in range(9)]
    palette_frames = [
        frame.quantize(colors=64, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        for frame in frames
    ]
    output = io.BytesIO()
    palette_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=palette_frames[1:],
        duration=[700] * 8 + [1600],
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output.getvalue()


def _validate_raster(name: str, data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if name == "terminal.png":
                _require(image.format == "PNG", "terminal_png_format_invalid")
                _require(image.width >= 1200 and image.height >= 1200, "terminal_png_size_invalid")
                _require(getattr(image, "n_frames", 1) == 1, "terminal_png_frames_invalid")
                _require(not image.info, "terminal_png_metadata_changed")
            else:
                _require(image.format == "GIF", "workflow_gif_format_invalid")
                _require(image.size == (1200, 675), "workflow_gif_size_invalid")
                _require(getattr(image, "n_frames", 1) == 9, "workflow_gif_frames_invalid")
                _require(image.info.get("loop") == 0, "workflow_gif_loop_invalid")
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError("raster_decode_failed") from exc


def _render_artifacts(
    receipt_bytes: bytes,
    network_observation: Mapping[str, Any],
) -> dict[str, bytes]:
    receipt = _validate_receipt(receipt_bytes)
    artifacts = {
        "architecture.svg": _render_architecture_svg(receipt),
        "contract-matrix.svg": _render_contract_matrix_svg(receipt, network_observation),
        "handoff-lab.txt": _render_transcript(receipt_bytes),
        "receipt.json": receipt_bytes,
        "terminal.png": _render_terminal_png(receipt_bytes),
        "terminal.svg": _render_terminal_svg(receipt_bytes),
        "tool-timeline.svg": _render_timeline_svg(receipt),
        "workflow-demo.gif": _render_workflow_gif(receipt),
    }
    _require(set(artifacts) == set(_ARTIFACT_MEDIA_TYPES), "artifact_set_changed")
    for name, data in artifacts.items():
        _require(0 < len(data) <= _MAX_ARTIFACT_BYTES, "artifact_size_invalid")
        if name.endswith(".svg"):
            evidence_core._validate_svg(data)
        elif name.endswith((".png", ".gif")):
            _validate_raster(name, data)
    return artifacts


def _runtime_manifest() -> dict[str, Any]:
    freetype_version = features.version("freetype2")
    zlib_version = features.version("zlib")
    _require(
        isinstance(freetype_version, str) and isinstance(zlib_version, str),
        "renderer_library_version_missing",
    )
    values = {
        "architecture": platform.machine(),
        "freetype_version": freetype_version,
        "k2do_distribution_version": importlib.metadata.version("k2do"),
        "operating_system": platform.system(),
        "pillow_version": importlib.metadata.version("Pillow"),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "zlib_version": zlib_version,
    }
    _require(
        all(bool(_SAFE_RUNTIME_RE.fullmatch(value)) for value in values.values()),
        "runtime_metadata_invalid",
    )
    _require(values["pillow_version"] == "12.3.0", "pillow_version_changed")
    return {
        **values,
        "application_dependencies_locked": False,
        "environment_claim": (
            "Pillow is exactly pinned for visual rendering; the application dependency set "
            "is not lock-reproduced"
        ),
        "font": "Pillow embedded default",
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
                "bootstrap_sha256": _sha256(_ISOLATED_BOOTSTRAP.encode()),
                "dependency_imports": "plain interpreter purelib/platlib paths; absolute paths excluded",
                "module": _ISOLATED_MODULE,
                "python_flags": ["-I", "-S", "-B"],
                "snapshot": "private materialization of committed HEAD; absolute path excluded",
            },
            "network_observation": dict(network_observation),
            "reader_command": list(_READER_COMMAND),
            "receipt_sha256": _sha256(artifacts["receipt.json"]),
            "runtime": _runtime_manifest(),
            "stdout_bytes": len(artifacts["receipt.json"]),
            "visual_derivation": (
                "SVG, PNG, and GIF bytes are deterministic functions of the captured receipt "
                "and bounded strace observation under the recorded renderer runtime"
            ),
        },
        "schema": _MANIFEST_SCHEMA,
        "source": {
            "commit": head,
            "files": {name: dict(sources[name]) for name in sorted(sources)},
            "tree": tree,
            "verification": (
                "all current committed k2do package blobs plus pyproject.toml; "
                "content identity required, no ancestry requirement"
            ),
        },
    }


def _strict_manifest(
    manifest_bytes: bytes,
    expected_source_paths: Sequence[str],
) -> dict[str, Any]:
    manifest = _json_without_duplicates(manifest_bytes, "manifest_json_invalid")
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
    for name, media_type in _ARTIFACT_MEDIA_TYPES.items():
        entry = artifacts[name]
        _require(
            isinstance(entry, dict)
            and set(entry) == {"bytes", "media_type", "sha256"}
            and isinstance(entry["bytes"], int)
            and 0 < entry["bytes"] <= _MAX_ARTIFACT_BYTES
            and entry["media_type"] == media_type
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
            "visual_derivation",
        }
        and capture["reader_command"] == list(_READER_COMMAND)
        and capture["receipt_sha256"] == artifacts["receipt.json"]["sha256"]
        and capture["stdout_bytes"] == artifacts["receipt.json"]["bytes"]
        and capture["visual_derivation"]
        == (
            "SVG, PNG, and GIF bytes are deterministic functions of the captured receipt "
            "and bounded strace observation under the recorded renderer runtime"
        ),
        "manifest_capture_invalid",
    )
    execution = capture["execution"]
    _require(
        isinstance(execution, dict)
        and execution
        == {
            "bootstrap_sha256": _sha256(_ISOLATED_BOOTSTRAP.encode()),
            "dependency_imports": "plain interpreter purelib/platlib paths; absolute paths excluded",
            "module": _ISOLATED_MODULE,
            "python_flags": ["-I", "-S", "-B"],
            "snapshot": "private materialization of committed HEAD; absolute path excluded",
        },
        "manifest_execution_invalid",
    )
    observation = capture["network_observation"]
    _require(
        isinstance(observation, dict)
        and set(observation)
        == {"communication_syscall_count", "scope", "status", "syscalls", "tool"}
        and observation["communication_syscall_count"] == 0
        and observation["scope"] == _NETWORK_OBSERVATION_SCOPE
        and observation["status"] == "no_communication_syscalls_observed"
        and observation["syscalls"] == list(evidence_core._COMMUNICATION_SYSCALLS)
        and isinstance(observation["tool"], str)
        and bool(_SAFE_RUNTIME_RE.fullmatch(observation["tool"])),
        "manifest_network_observation_invalid",
    )
    runtime = capture["runtime"]
    _require(
        isinstance(runtime, dict)
        and set(runtime)
        == {
            "application_dependencies_locked",
            "architecture",
            "environment_claim",
            "font",
            "freetype_version",
            "k2do_distribution_version",
            "operating_system",
            "pillow_version",
            "python_implementation",
            "python_version",
            "zlib_version",
        }
        and runtime["application_dependencies_locked"] is False
        and runtime["pillow_version"] == "12.3.0"
        and runtime["font"] == "Pillow embedded default"
        and runtime["environment_claim"]
        == (
            "Pillow is exactly pinned for visual rendering; the application dependency set "
            "is not lock-reproduced"
        )
        and all(
            isinstance(runtime[key], str) and bool(_SAFE_RUNTIME_RE.fullmatch(runtime[key]))
            for key in (
                "architecture",
                "freetype_version",
                "k2do_distribution_version",
                "operating_system",
                "python_implementation",
                "python_version",
                "zlib_version",
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
        == (
            "all current committed k2do package blobs plus pyproject.toml; "
            "content identity required, no ancestry requirement"
        )
        and isinstance(source["files"], dict)
        and set(source["files"]) == set(expected_source_paths),
        "manifest_source_invalid",
    )
    for entry in source["files"].values():
        _require(
            isinstance(entry, dict)
            and set(entry) == {"bytes", "git_blob", "git_mode", "sha256"}
            and isinstance(entry["bytes"], int)
            and 0 < entry["bytes"] <= _MAX_SOURCE_BYTES
            and isinstance(entry["git_blob"], str)
            and bool(_HEX_OBJECT_RE.fullmatch(entry["git_blob"]))
            and entry["git_mode"] in {"100644", "100755"}
            and isinstance(entry["sha256"], str)
            and bool(_HEX_SHA256_RE.fullmatch(entry["sha256"])),
            "manifest_source_entry_invalid",
        )
    return manifest


def _assert_publication_state(
    repository: Path,
    *,
    head: str,
    tree: str,
    sources: Mapping[str, Mapping[str, Any]],
    source_paths: Sequence[str],
    stage_name: str,
    file_names: Sequence[str],
) -> None:
    current_head = (
        evidence_core._run_git(
            repository,
            ["rev-parse", "--verify", "HEAD"],
        )
        .decode()
        .strip()
    )
    current_tree = (
        evidence_core._run_git(
            repository,
            ["rev-parse", "--verify", "HEAD^{tree}"],
        )
        .decode()
        .strip()
    )
    _require(current_head == head and current_tree == tree, "head_changed_before_publication")
    _require(
        _source_inventory(repository, source_paths) == sources, "source_changed_before_publication"
    )
    status = evidence_core._run_git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    expected = b"".join(f"?? docs/{stage_name}/{name}\0".encode() for name in sorted(file_names))
    _require(status == expected, "worktree_changed_before_publication")


def _publish_artifacts(
    repository: Path,
    files: Mapping[str, bytes],
    *,
    pre_publish: Callable[[str], None],
) -> None:
    root_fd = evidence_core._open_repository_root(repository)
    docs_fd = -1
    stage_fd = -1
    stage_name = f".{_OUTPUT_DIRECTORY}.stage-{os.getpid()}-{secrets.token_hex(8)}"
    stage_created = False
    try:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        docs_fd = os.open("docs", flags, dir_fd=root_fd)
        try:
            os.stat(_OUTPUT_DIRECTORY, dir_fd=docs_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EvidenceError("evidence_directory_already_exists")
        os.mkdir(stage_name, 0o700, dir_fd=docs_fd)
        stage_created = True
        stage_fd = os.open(stage_name, flags, dir_fd=docs_fd)
        _require(stat.S_ISDIR(os.fstat(stage_fd).st_mode), "staging_leaf_not_directory")
        for name in sorted(files):
            _require("/" not in name and name not in {".", ".."}, "artifact_name_invalid")
            file_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            file_fd = os.open(name, file_flags, 0o644, dir_fd=stage_fd)
            try:
                os.fchmod(file_fd, 0o644)
                evidence_core._write_all(file_fd, files[name])
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
        os.fsync(stage_fd)
        pre_publish(stage_name)
        os.close(stage_fd)
        stage_fd = -1
        evidence_core._rename_noreplace(
            docs_fd,
            stage_name,
            docs_fd,
            _OUTPUT_DIRECTORY,
        )
        stage_created = False
        os.fsync(docs_fd)
    except OSError as exc:
        raise EvidenceError("artifact_publication_io_failed") from exc
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        if stage_created and docs_fd >= 0:
            cleanup_fd = -1
            try:
                cleanup_fd = os.open(
                    stage_name,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=docs_fd,
                )
                for name in sorted(files):
                    try:
                        metadata = os.stat(name, dir_fd=cleanup_fd, follow_symlinks=False)
                        if stat.S_ISREG(metadata.st_mode):
                            os.unlink(name, dir_fd=cleanup_fd)
                    except OSError:
                        pass
            except OSError:
                pass
            finally:
                if cleanup_fd >= 0:
                    os.close(cleanup_fd)
            try:
                os.rmdir(stage_name, dir_fd=docs_fd)
            except OSError:
                pass
        if docs_fd >= 0:
            os.close(docs_fd)
        os.close(root_fd)


def _read_published_files(repository: Path) -> dict[str, bytes]:
    root_fd = evidence_core._open_repository_root(repository)
    docs_fd = -1
    evidence_fd = -1
    try:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        docs_fd = os.open("docs", flags, dir_fd=root_fd)
        evidence_fd = os.open(_OUTPUT_DIRECTORY, flags, dir_fd=docs_fd)
        expected = set(_ARTIFACT_MEDIA_TYPES) | {"manifest.json"}
        _require(set(os.listdir(evidence_fd)) == expected, "evidence_file_set_changed")
        files: dict[str, bytes] = {}
        for name in sorted(expected):
            file_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=evidence_fd,
            )
            try:
                metadata = os.fstat(file_fd)
                _require(
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_mode & 0o111 == 0
                    and metadata.st_mode & 0o022 == 0
                    and 0 < metadata.st_size <= _MAX_ARTIFACT_BYTES,
                    "evidence_file_mode_or_size_invalid",
                )
                chunks: list[bytes] = []
                remaining = _MAX_ARTIFACT_BYTES + 1
                while remaining > 0:
                    chunk = os.read(file_fd, min(64 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                _require(len(data) == metadata.st_size, "artifact_size_changed_during_read")
                files[name] = data
            finally:
                os.close(file_fd)
        return files
    except OSError as exc:
        raise EvidenceError("evidence_directory_not_safe") from exc
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
    """Capture and atomically publish a new absent handoff evidence bundle."""

    repository = (repository or _repository_root()).resolve()
    head, tree = evidence_core._assert_clean_committed_head(repository)
    source_paths = _source_paths(repository, head)
    sources_before = _source_inventory(repository, source_paths)
    if capture is None:
        with evidence_core._committed_snapshot(repository, head) as snapshot:
            receipt_bytes, network_observation = _capture_from_snapshot(repository, snapshot)
    else:
        receipt_bytes, network_observation = capture(repository)
    _validate_receipt(receipt_bytes)
    artifacts = _render_artifacts(receipt_bytes, network_observation)
    _require(
        _source_inventory(repository, source_paths) == sources_before,
        "source_changed_during_capture",
    )
    current_head, current_tree = evidence_core._assert_clean_committed_head(repository)
    _require(current_head == head and current_tree == tree, "head_changed_during_capture")
    manifest = _build_manifest(
        head=head,
        tree=tree,
        sources=sources_before,
        artifacts=artifacts,
        network_observation=network_observation,
    )
    files = dict(artifacts)
    files["manifest.json"] = _canonical_json(manifest)
    _strict_manifest(files["manifest.json"], source_paths)

    def verify(stage_name: str) -> None:
        _assert_publication_state(
            repository,
            head=head,
            tree=tree,
            sources=sources_before,
            source_paths=source_paths,
            stage_name=stage_name,
            file_names=files,
        )

    _publish_artifacts(repository, files, pre_publish=verify)
    return {
        "artifact_count": len(artifacts),
        "receipt_sha256": _sha256(receipt_bytes),
        "schema": _SCHEMA,
        "source_commit": head,
        "source_file_count": len(source_paths),
        "status": "recorded",
    }


def check(
    repository: Path | None = None,
    *,
    capture: CaptureFunction | None = None,
    recapture: bool | None = None,
) -> dict[str, Any]:
    """Verify bundle bytes, renderers, source blobs, and an optional recapture."""

    repository = (repository or _repository_root()).resolve()
    head = evidence_core._run_git(repository, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    source_paths = _source_paths(repository, head)
    files = _read_published_files(repository)
    manifest = _strict_manifest(files["manifest.json"], source_paths)
    capture_object = evidence_core._verify_capture_object(repository, manifest["source"])
    current_sources = _source_inventory(repository, source_paths)
    _require(current_sources == manifest["source"]["files"], "source_identity_mismatch")
    for name, entry in manifest["artifacts"].items():
        data = files[name]
        _require(
            len(data) == entry["bytes"] and _sha256(data) == entry["sha256"],
            "artifact_identity_mismatch",
        )
    receipt_bytes = files["receipt.json"]
    _validate_receipt(receipt_bytes)
    expected = _render_artifacts(receipt_bytes, manifest["capture"]["network_observation"])
    for name, rendered in expected.items():
        _require(files[name] == rendered, "artifact_rendering_mismatch")

    if recapture is None:
        recapture = platform.system() == "Linux" and os.path.isfile(evidence_core._STRACE_PATH)
    fresh_capture = "not_available"
    if recapture:
        captured, observation = (capture or _capture_receipt)(repository)
        _require(captured == receipt_bytes, "fresh_receipt_mismatch")
        _require(
            observation["communication_syscall_count"] == 0
            and observation["status"] == manifest["capture"]["network_observation"]["status"]
            and observation["syscalls"] == manifest["capture"]["network_observation"]["syscalls"],
            "fresh_network_observation_mismatch",
        )
        fresh_capture = "verified"
    _require(
        _source_inventory(repository, source_paths) == current_sources,
        "source_changed_during_check",
    )
    return {
        "artifact_count": len(_ARTIFACT_MEDIA_TYPES),
        "capture_object": capture_object,
        "fresh_capture": fresh_capture,
        "receipt_sha256": _sha256(receipt_bytes),
        "schema": _SCHEMA,
        "source_file_count": len(source_paths),
        "status": "verified",
    }


def _write_result(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_canonical_json(value).decode())


def main(argv: Sequence[str] | None = None) -> int:
    """Strict two-command CLI that never exposes private exception details."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        sys.stdout.write(
            "usage: python -m k2do.labs.handoff_evidence_recorder {record|check}\n"
            "Record or verify source-bound routed handoff evidence.\n"
        )
        return 0
    if args not in (["record"], ["check"]):
        _write_result(
            sys.stderr,
            {"failure": "invalid_invocation", "schema": _SCHEMA, "status": "failed"},
        )
        return 2
    try:
        result = record() if args == ["record"] else check()
    except EvidenceError as exc:
        _write_result(
            sys.stderr,
            {"failure": exc.code, "schema": _SCHEMA, "status": "failed"},
        )
        return 1
    except Exception:
        _write_result(
            sys.stderr,
            {"failure": "internal_error", "schema": _SCHEMA, "status": "failed"},
        )
        return 1
    _write_result(sys.stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
