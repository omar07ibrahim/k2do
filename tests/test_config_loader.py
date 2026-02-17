import json
from pathlib import Path

from k2do.config.loader import convert_keys, convert_to_camel, load_config


def test_convert_keys_preserves_header_and_env_map_keys() -> None:
    raw = {
        "providers": {
            "k2Think": {
                "apiKey": "x",
                "extraHeaders": {
                    "APP-Code": "123",
                    "X-Api-Key": "abc",
                },
            }
        },
        "tools": {
            "mcpServers": {
                "srv-1": {
                    "command": "echo",
                    "env": {
                        "OPENAI_API_KEY": "k",
                        "My-Key": "v",
                    },
                }
            }
        },
    }

    converted = convert_keys(raw)
    assert converted["providers"]["k2_think"]["extra_headers"]["APP-Code"] == "123"
    assert converted["providers"]["k2_think"]["extra_headers"]["X-Api-Key"] == "abc"
    assert converted["tools"]["mcp_servers"]["srv-1"]["env"]["OPENAI_API_KEY"] == "k"
    assert converted["tools"]["mcp_servers"]["srv-1"]["env"]["My-Key"] == "v"

    roundtrip = convert_to_camel(converted)
    assert roundtrip["providers"]["k2Think"]["extraHeaders"]["APP-Code"] == "123"
    assert roundtrip["tools"]["mcpServers"]["srv-1"]["env"]["OPENAI_API_KEY"] == "k"


def test_load_config_env_overrides_file(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "model": "k2-think-v2/LLM360/K2-Think-V2",
                        "maxTokens": 4096,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "K2DO_AGENTS__DEFAULTS__MODEL",
        '"k2-v2-instruct/LLM360/K2-V2-Instruct"',
    )
    monkeypatch.setenv("K2DO_AGENTS__DEFAULTS__MAX_TOKENS", "16384")

    cfg = load_config(cfg_path)
    assert cfg.agents.defaults.model == "k2-v2-instruct/LLM360/K2-V2-Instruct"
    assert cfg.agents.defaults.max_tokens == 16384


def test_env_override_preserves_dynamic_map_keys(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("K2DO_TOOLS__MCP_SERVERS__srv_1__COMMAND", "echo")
    monkeypatch.setenv("K2DO_TOOLS__MCP_SERVERS__srv_1__ENV__OPENAI_API_KEY", "k")

    cfg = load_config(cfg_path)
    assert "srv_1" in cfg.tools.mcp_servers
    assert cfg.tools.mcp_servers["srv_1"].command == "echo"
    assert cfg.tools.mcp_servers["srv_1"].env["OPENAI_API_KEY"] == "k"
