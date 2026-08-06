"""Fail-closed transport configuration tests for MCP servers."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from k2do.config.loader import load_config, save_config
from k2do.config.schema import MCPServerConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"command": "python", "url": "https://mcp.example.test"},
    ],
)
def test_transport_requires_exactly_one_command_or_url(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="exactly one transport"):
        MCPServerConfig(**kwargs)


@pytest.mark.parametrize(
    "extra_field",
    [
        {"connect_timeout_second": 5.0},
        {"protcol_mode": "legacy"},
    ],
)
def test_unknown_fields_are_rejected(extra_field: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MCPServerConfig(command="python", **extra_field)


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.example.test/rpc",
        "http://192.168.1.20/rpc",
        "http://10.0.0.4/rpc",
    ],
)
def test_cleartext_remote_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="Remote MCP URLs must use HTTPS"):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://user@mcp.example.test/rpc",
        "https://user:password@mcp.example.test/rpc",
        "http://token@localhost/rpc",
    ],
)
def test_urls_with_user_information_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="must not include user information"):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.test\\@127.0.0.1/mcp",
        "https://mcp.example.test/mcp\\status",
    ],
)
def test_urls_with_backslashes_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="must not contain backslashes"):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.test/mcp#credentials",
        "https://mcp.example.test/mcp#",
    ],
)
def test_url_fragments_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="must not include a fragment"):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "query",
    [
        "token=value",
        "api_key=value",
        "API_KEY=value",
        "key=value",
        "secret=value",
        "password=value",
        "access_token=value",
        "client_secret=value",
        "tenant=public",
        "",
    ],
)
def test_url_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValidationError, match="must not include a query string"):
        MCPServerConfig(url=f"https://mcp.example.test/mcp?{query}")


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            "https://user%40example.test@mcp.example.test/mcp",
            "must not include user information",
        ),
        (
            "https://mcp.example.test%40attacker.example/mcp",
            "authority must not use percent encoding",
        ),
        (
            "https://%6dcp.example.test/mcp",
            "authority must not use percent encoding",
        ),
    ],
)
def test_percent_encoded_authority_confusion_is_rejected(
    url: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.test:/mcp",
        "https://mcp.example.test:0/mcp",
        "https://mcp.example.test:not-a-port/mcp",
        "https://mcp.example.test:65536/mcp",
        "https://mcp.example.test:443:80/mcp",
        "https://[::1]:/mcp",
        "https://[::1/mcp",
        "https://[::1]extra/mcp",
    ],
)
def test_malformed_authorities_and_ports_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.test/mcp\x7f",
        "https://mcp.example.test/mcp\u0085",
        "https://mcp.example.test/mcp\u200b",
        "https://mcp.ex\N{CYRILLIC SMALL LETTER A}mple.test/mcp",
    ],
)
def test_unicode_and_control_characters_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/mcp",
        "http://127.0.0.1/mcp",
        "http://127.255.255.254/mcp",
        "http://[::1]:8080/mcp",
        "https://mcp.example.test/rpc",
    ],
)
def test_local_http_and_remote_https_urls_are_accepted(url: str) -> None:
    assert MCPServerConfig(url=url).url == url


def test_camel_case_config_survives_load_and_save_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text(
        json.dumps(
            {
                "tools": {
                    "mcpServers": {
                        "local-fixture": {
                            "url": "http://[::1]:8765/mcp",
                            "protocolMode": "legacy",
                            "connectTimeoutSeconds": 4.5,
                            "callTimeoutSeconds": 8.5,
                            "maxOutputBytes": 4096,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(source)
    server = config.tools.mcp_servers["local-fixture"]
    assert server.protocol_mode == "legacy"
    assert server.connect_timeout_seconds == 4.5
    assert server.call_timeout_seconds == 8.5
    assert server.max_output_bytes == 4096

    save_config(config, destination)
    saved_server = json.loads(destination.read_text(encoding="utf-8"))["tools"][
        "mcpServers"
    ]["local-fixture"]
    assert saved_server["protocolMode"] == "legacy"
    assert saved_server["connectTimeoutSeconds"] == 4.5
    assert saved_server["callTimeoutSeconds"] == 8.5
    assert saved_server["maxOutputBytes"] == 4096
