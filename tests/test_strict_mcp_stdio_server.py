from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

import pytest

from k2do.labs import strict_mcp_stdio_server as fixture


def _meta() -> dict:
    return {
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "offline-test-client",
            "version": "1.0.0",
        },
        "io.modelcontextprotocol/protocolVersion": fixture.PROTOCOL_VERSION,
    }


def _request(request_id: str, method: str, **params: object) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {"_meta": _meta(), **params},
    }


def _server(behavior: str = "healthy") -> tuple[fixture.StrictMCPServer, list[tuple[str, str | None]]]:
    events: list[tuple[str, str | None]] = []
    server = fixture.StrictMCPServer(
        behavior,
        fault_sentinel=fixture.FAULT_SENTINEL,
        audit=lambda event, method=None: events.append((event, method)),
    )
    return server, events


def test_modern_discovery_and_catalog_are_closed_and_cache_explicit() -> None:
    server, events = _server()

    discovery = server.handle(_request("discover-1", "server/discover"))
    assert discovery == {
        "jsonrpc": "2.0",
        "id": "discover-1",
        "result": {
            "_meta": fixture.SERVER_STAMP,
            "cacheScope": "public",
            "capabilities": {"tools": {}},
            "resultType": "complete",
            "supportedVersions": ["2026-07-28"],
            "ttlMs": 0,
        },
    }

    listing = server.handle(_request("list-1", "tools/list"))
    assert listing is not None
    result = listing["result"]
    assert result["cacheScope"] == "public"
    assert result["resultType"] == "complete"
    assert result["ttlMs"] == 0
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool["name"] == fixture.TOOL_NAME
    assert tool["inputSchema"]["additionalProperties"] is False
    assert tool["outputSchema"]["additionalProperties"] is False
    assert events == [
        ("request_received", "server/discover"),
        ("discovery_completed", "server/discover"),
        ("request_received", "tools/list"),
        ("catalog_listed", "tools/list"),
    ]


def test_call_error_is_private_and_next_call_recovers() -> None:
    server, events = _server("call_error_then_recover")
    first = server.handle(
        _request(
            "call-1",
            "tools/call",
            name=fixture.TOOL_NAME,
            arguments={"value": 7},
        )
    )
    second = server.handle(
        _request(
            "call-2",
            "tools/call",
            name=fixture.TOOL_NAME,
            arguments={"value": 7},
        )
    )

    assert first is not None
    assert first["error"]["code"] == -32603
    assert fixture.FAULT_SENTINEL in first["error"]["message"]
    assert second is not None
    assert second["result"]["structuredContent"] == {"accepted": 7}
    assert events == [
        ("request_received", "tools/call"),
        ("call_failed", "tools/call"),
        ("request_received", "tools/call"),
        ("call_completed", "tools/call"),
    ]


def test_held_call_accepts_only_its_cancellation_notification() -> None:
    server, events = _server("call_hang")
    assert (
        server.handle(
            _request(
                "held-1",
                "tools/call",
                name=fixture.TOOL_NAME,
                arguments={"value": 3},
            )
        )
        is None
    )

    assert (
        server.handle(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "held-1", "reason": "caller cancelled"},
            }
        )
        is None
    )
    assert events[-2:] == [
        ("request_received", "notifications/cancelled"),
        ("cancellation_observed", "notifications/cancelled"),
    ]

    assert (
        server.handle(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "held-1"},
            }
        )
        is None
    )
    assert events[-1] == ("cancellation_ignored", "notifications/cancelled")


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"jsonrpc":"2.0","jsonrpc":"2.0"}\n', "duplicate_object_key"),
        (b'{"value":NaN}\n', "non_finite_number"),
        (b"[]\n", "request_not_object"),
        (b"{}", "invalid_frame"),
        (b"\xff\n", "invalid_json"),
    ],
)
def test_decoder_rejects_ambiguous_or_non_json_frames(raw: bytes, code: str) -> None:
    with pytest.raises(fixture.JSONParseError, match=code):
        fixture.decode_message(raw)


def test_decoder_accepts_exact_canonical_frame() -> None:
    message = _request("stable-id", "server/discover")
    raw = (fixture.canonical_json(message) + "\n").encode("utf-8")
    assert fixture.decode_message(raw) == message
    assert json.loads(raw) == message


@pytest.mark.parametrize(
    "mutation",
    [
        {"id": True},
        {"extra": "field"},
        {"params": {}},
        {"params": {"_meta": {}}},
    ],
)
def test_server_rejects_malformed_envelopes_without_reflection(mutation: dict) -> None:
    server, _events = _server()
    message = _request("request-1", "server/discover")
    message.update(mutation)
    with pytest.raises(fixture.ProtocolViolationError) as caught:
        server.handle(message)
    rendered = str(caught.value)
    assert "request-1" not in rendered
    assert "offline-test-client" not in rendered


def test_unknown_method_returns_fixed_protocol_error() -> None:
    server, events = _server()
    response = server.handle(_request("unknown-1", "unknown/method"))
    assert response == {
        "jsonrpc": "2.0",
        "id": "unknown-1",
        "error": {"code": -32601, "message": "Method not found"},
    }
    assert events == [("request_received", None)]


def test_main_rejects_arguments_without_echo(capsys: pytest.CaptureFixture[str]) -> None:
    rejected = "private-argument"
    assert fixture.main([rejected]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert rejected not in captured.out + captured.err


def _start_subprocess(
    tmp_path: Path,
    behavior: str,
) -> tuple[subprocess.Popen[bytes], socket.socket, str]:
    socket_path = tmp_path / "audit.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(3)
    listener.bind(str(socket_path))
    listener.listen(1)
    token = "private-audit-credential-never-on-stdio"
    environment = dict(os.environ)
    environment.update(
        {
            "K2DO_MCP_LAB_AUDIT_SOCKET": str(socket_path),
            "K2DO_MCP_LAB_AUDIT_TOKEN": token,
            "K2DO_MCP_LAB_BEHAVIOR": behavior,
            "K2DO_MCP_LAB_INSTANCE": "subprocess-test",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "k2do.labs.strict_mcp_stdio_server"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        audit, _address = listener.accept()
    except BaseException:
        process.kill()
        process.wait(timeout=3)
        raise
    finally:
        listener.close()
    audit.settimeout(3)
    return process, audit, token


def _write_request(stream: BinaryIO, request_id: int, method: str, **params: object) -> None:
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {"_meta": _meta(), **params},
    }
    stream.write((fixture.canonical_json(request) + "\n").encode("utf-8"))
    stream.flush()


def _read_audit(audit: socket.socket) -> list[dict]:
    raw = b""
    while True:
        chunk = audit.recv(4096)
        if not chunk:
            break
        raw += chunk
    return [json.loads(line) for line in raw.splitlines()]


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def test_real_stdio_subprocess_negotiates_calls_and_exits_on_eof(tmp_path: Path) -> None:
    process, audit, token = _start_subprocess(tmp_path, "healthy")
    responses: list[bytes] = []
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        _write_request(process.stdin, 1, "server/discover")
        responses.append(process.stdout.readline())
        assert json.loads(responses[-1])["result"]["supportedVersions"] == [
            fixture.PROTOCOL_VERSION
        ]

        _write_request(process.stdin, 2, "tools/list")
        responses.append(process.stdout.readline())
        assert json.loads(responses[-1])["result"]["tools"][0]["name"] == fixture.TOOL_NAME

        _write_request(
            process.stdin,
            3,
            "tools/call",
            name=fixture.TOOL_NAME,
            arguments={"value": 9},
        )
        responses.append(process.stdout.readline())
        assert json.loads(responses[-1])["result"]["structuredContent"] == {"accepted": 9}

        process.stdin.close()
        assert process.wait(timeout=3) == 0
        assert process.stderr.read() == b""
        assert token.encode("utf-8") not in b"".join(responses)

        audit_events = _read_audit(audit)
        assert [event["event"] for event in audit_events][-2:] == [
            "stdin_eof",
            "process_exit",
        ]
        assert [event["ordinal"] for event in audit_events] == list(
            range(1, len(audit_events) + 1)
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        audit.close()
        _close_process_streams(process)


def test_real_stdio_subprocess_maps_invalid_json_to_parse_error(tmp_path: Path) -> None:
    process, audit, token = _start_subprocess(tmp_path, "healthy")
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(b'{"jsonrpc":\n')
        process.stdin.flush()
        response = process.stdout.readline()
        assert json.loads(response) == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        assert process.wait(timeout=3) == 1
        assert process.stderr.read() == b""
        assert token.encode("utf-8") not in response
        assert [event["event"] for event in _read_audit(audit)][-2:] == [
            "fixture_failed",
            "process_exit",
        ]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        audit.close()
        _close_process_streams(process)
