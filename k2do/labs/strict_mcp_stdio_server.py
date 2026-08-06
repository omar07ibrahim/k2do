"""Independent strict MCP stdio fixture for the fault laboratory.

The server intentionally uses only the standard library. K2DO connects through
its real pinned SDK client, while lifecycle details travel on a private Unix
socket and never enter the public receipt.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from collections.abc import Callable
from typing import Any

PROTOCOL_VERSION = "2026-07-28"
TOOL_NAME = "resilience_probe"
FAULT_SENTINEL = "k2do-fixture-raw-fault-sentinel"
MAX_LINE_BYTES = 64 * 1024
AUDIT_TIMEOUT_SECONDS = 1.0
BEHAVIORS = frozenset(
    {"healthy", "list_failure", "call_error_then_recover", "startup_hang", "call_hang"}
)
AUDIT_EVENTS = frozenset(
    {
        "booted",
        "request_received",
        "discovery_completed",
        "discovery_held",
        "catalog_listed",
        "catalog_failed",
        "call_completed",
        "call_failed",
        "call_held",
        "cancellation_observed",
        "cancellation_ignored",
        "stdin_eof",
        "fixture_failed",
        "process_exit",
    }
)
METHODS = frozenset(
    {"server/discover", "tools/list", "tools/call", "notifications/cancelled"}
)
SERVER_STAMP = {
    "io.modelcontextprotocol/serverInfo": {
        "name": "k2do-strict-fault-fixture",
        "version": "1.0.0",
    }
}
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["value"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["accepted"],
    "additionalProperties": False,
}


class ProtocolViolationError(ValueError):
    """A malformed frame or request, represented without hostile input."""


class JSONParseError(ProtocolViolationError):
    """A frame that cannot be interpreted as one unambiguous JSON value."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise JSONParseError("duplicate_object_key")
        value[key] = child
    return value


def _reject_constant(_value: str) -> None:
    raise JSONParseError("non_finite_number")


def decode_message(raw: bytes) -> dict[str, Any]:
    """Decode one bounded JSON-RPC line with duplicate-key rejection."""
    if not raw or len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
        raise JSONParseError("invalid_frame")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except JSONParseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise JSONParseError("invalid_json") from exc
    if type(value) is not dict:
        raise ProtocolViolationError("request_not_object")
    return value


def _valid_request_id(value: object) -> bool:
    if type(value) is int:
        return True
    if type(value) is not str:
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return 0 < len(encoded) <= 128


def _error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _result(request_id: str | int, value: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def _validate_modern_meta(params: dict[str, Any]) -> None:
    meta = params.get("_meta")
    if type(meta) is not dict:
        raise ProtocolViolationError("missing_request_meta")
    if meta.get("io.modelcontextprotocol/protocolVersion") != PROTOCOL_VERSION:
        raise ProtocolViolationError("unsupported_protocol_version")
    if type(meta.get("io.modelcontextprotocol/clientCapabilities")) is not dict:
        raise ProtocolViolationError("missing_client_capabilities")
    client_info = meta.get("io.modelcontextprotocol/clientInfo")
    if client_info is not None and type(client_info) is not dict:
        raise ProtocolViolationError("invalid_client_info")


class StrictMCPServer:
    """State machine implementing the exact modern MCP surface used by the lab."""

    def __init__(
        self,
        behavior: str,
        *,
        fault_sentinel: str,
        audit: Callable[[str, str | None], None],
    ) -> None:
        if behavior not in BEHAVIORS:
            raise ValueError("unsupported_fixture_behavior")
        if fault_sentinel != FAULT_SENTINEL:
            raise ValueError("invalid_fault_sentinel")
        self.behavior = behavior
        self._fault_sentinel = fault_sentinel
        self._audit = audit
        self._call_count = 0
        self._held_call_ids: set[str | int] = set()

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Validate and handle one request or cancellation notification."""
        if message.get("jsonrpc") != "2.0" or type(message.get("method")) is not str:
            raise ProtocolViolationError("invalid_jsonrpc_envelope")
        method = message["method"]
        is_notification = "id" not in message
        allowed_keys = {"jsonrpc", "method", "params"}
        if not is_notification:
            allowed_keys.add("id")
        if set(message) != allowed_keys or type(message.get("params")) is not dict:
            raise ProtocolViolationError("invalid_jsonrpc_shape")

        self._audit("request_received", method if method in METHODS else None)
        if method == "notifications/cancelled":
            if not is_notification:
                raise ProtocolViolationError("cancellation_must_be_notification")
            return self._handle_cancellation(message["params"])

        request_id = message.get("id")
        if not _valid_request_id(request_id):
            raise ProtocolViolationError("invalid_request_id")
        assert type(request_id) in {str, int}
        params = message["params"]
        _validate_modern_meta(params)

        if method == "server/discover":
            return self._discover(request_id)
        if method == "tools/list":
            return self._list_tools(request_id, params)
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return _error(request_id, -32601, "Method not found")

    def _discover(self, request_id: str | int) -> dict[str, Any] | None:
        if self.behavior == "startup_hang":
            self._audit("discovery_held", "server/discover")
            return None
        self._audit("discovery_completed", "server/discover")
        return _result(
            request_id,
            {
                "_meta": SERVER_STAMP,
                "cacheScope": "public",
                "capabilities": {"tools": {}},
                "resultType": "complete",
                "supportedVersions": [PROTOCOL_VERSION],
                "ttlMs": 0,
            },
        )

    def _list_tools(self, request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        if set(params) != {"_meta"}:
            raise ProtocolViolationError("unexpected_catalog_params")
        if self.behavior == "list_failure":
            self._audit("catalog_failed", "tools/list")
            return _error(
                request_id,
                -32603,
                "Synthetic catalog fault: " + self._fault_sentinel,
            )
        self._audit("catalog_listed", "tools/list")
        return _result(
            request_id,
            {
                "_meta": SERVER_STAMP,
                "cacheScope": "public",
                "resultType": "complete",
                "tools": [
                    {
                        "name": TOOL_NAME,
                        "description": "Verify a bounded integer through the strict fixture.",
                        "inputSchema": INPUT_SCHEMA,
                        "outputSchema": OUTPUT_SCHEMA,
                    }
                ],
                "ttlMs": 0,
            },
        )

    def _call_tool(
        self,
        request_id: str | int,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        if set(params) != {"_meta", "arguments", "name"}:
            raise ProtocolViolationError("unexpected_call_params")
        arguments = params.get("arguments")
        if params.get("name") != TOOL_NAME or type(arguments) is not dict:
            raise ProtocolViolationError("invalid_tool_call")
        if set(arguments) != {"value"}:
            raise ProtocolViolationError("invalid_tool_arguments")
        value = arguments["value"]
        if type(value) is not int or not 0 <= value <= 100:
            raise ProtocolViolationError("invalid_tool_value")

        self._call_count += 1
        if self.behavior == "call_error_then_recover" and self._call_count == 1:
            self._audit("call_failed", "tools/call")
            return _error(
                request_id,
                -32603,
                "Synthetic call fault: " + self._fault_sentinel,
            )
        if self.behavior == "call_hang":
            self._held_call_ids.add(request_id)
            self._audit("call_held", "tools/call")
            return None

        self._audit("call_completed", "tools/call")
        return _result(
            request_id,
            {
                "_meta": SERVER_STAMP,
                "content": [{"type": "text", "text": "accepted"}],
                "isError": False,
                "resultType": "complete",
                "structuredContent": {"accepted": value},
            },
        )

    def _handle_cancellation(self, params: dict[str, Any]) -> None:
        if "requestId" not in params or not set(params) <= {"requestId", "reason"}:
            raise ProtocolViolationError("invalid_cancellation_params")
        request_id = params["requestId"]
        reason = params.get("reason")
        if reason is not None and type(reason) is not str:
            raise ProtocolViolationError("invalid_cancellation_reason")
        if request_id not in self._held_call_ids:
            self._audit("cancellation_ignored", "notifications/cancelled")
            return None
        self._held_call_ids.remove(request_id)
        self._audit("cancellation_observed", "notifications/cancelled")
        return None


class _UnixAuditSink:
    """Authenticated lifecycle channel that never writes stdout or stderr."""

    def __init__(self, path: str, token: str, instance: str) -> None:
        if (
            not path
            or not token
            or not instance
            or len(instance) > 64
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in instance)
        ):
            raise ValueError("invalid_audit_configuration")
        self._token = token
        self._instance = instance
        self._ordinal = 0
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(AUDIT_TIMEOUT_SECONDS)
        self._socket.connect(path)

    def record(self, event: str, method: str | None = None) -> None:
        if event not in AUDIT_EVENTS or (method is not None and method not in METHODS):
            raise ValueError("invalid_audit_event")
        self._ordinal += 1
        payload: dict[str, Any] = {
            "event": event,
            "instance": self._instance,
            "ordinal": self._ordinal,
            "token": self._token,
        }
        if event == "booted":
            payload["pid"] = os.getpid()
        if method is not None:
            payload["method"] = method
        self._socket.sendall((canonical_json(payload) + "\n").encode("utf-8"))

    def close(self) -> None:
        self._socket.close()


def _send(message: dict[str, Any]) -> None:
    sys.stdout.buffer.write((canonical_json(message) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _serve(server: StrictMCPServer, audit: _UnixAuditSink) -> int:
    while True:
        raw = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if raw == b"":
            audit.record("stdin_eof")
            return 0
        request_id: str | int | None = None
        try:
            message = decode_message(raw)
            candidate_id = message.get("id")
            if _valid_request_id(candidate_id):
                request_id = candidate_id  # type: ignore[assignment]
            response = server.handle(message)
        except JSONParseError:
            audit.record("fixture_failed")
            _send(_error(None, -32700, "Parse error"))
            return 1
        except ProtocolViolationError:
            audit.record("fixture_failed")
            _send(_error(request_id, -32600, "Invalid Request"))
            return 1
        if response is not None:
            _send(response)


def main(argv: list[str] | None = None) -> int:
    """Run without reflecting rejected arguments or private failures."""
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        return 2

    sink: _UnixAuditSink | None = None
    try:
        sink = _UnixAuditSink(
            os.environ["K2DO_MCP_LAB_AUDIT_SOCKET"],
            os.environ["K2DO_MCP_LAB_AUDIT_TOKEN"],
            os.environ["K2DO_MCP_LAB_INSTANCE"],
        )
        sink.record("booted")
        server = StrictMCPServer(
            os.environ["K2DO_MCP_LAB_BEHAVIOR"],
            fault_sentinel=FAULT_SENTINEL,
            audit=sink.record,
        )
        return _serve(server, sink)
    except Exception:
        if sink is not None:
            try:
                sink.record("fixture_failed")
            except Exception:
                pass
        return 1
    finally:
        if sink is not None:
            try:
                sink.record("process_exit")
            except Exception:
                pass
            try:
                sink.close()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
