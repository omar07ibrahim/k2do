"""Configuration loading for K2DO."""

import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from k2do.config.schema import Config


def get_config_path() -> Path:
    return Path.home() / ".k2do" / "config.json"


def get_data_dir() -> Path:
    from k2do.utils.helpers import get_data_path
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    path = config_path or get_config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    normalized = convert_keys(data)
    merged = _apply_env_overrides(normalized)
    return Config.model_validate(merged)


def save_config(config: Config, config_path: Path | None = None) -> None:
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    data = convert_to_camel(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Normalize provider blocks for K2 Think + K2 Instruct.
    providers = data.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        data["providers"] = providers
    models = data.get("models", {})
    model_providers = models.get("providers", {}) if isinstance(models, dict) else {}

    # Accept external-style provider keys from user config snippets.
    provider_aliases: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("k2Think", "k2-think-v2", ("k2Think", "k2_think", "k2-think-v2")),
        ("k2Instruct", "k2-v2-instruct", ("k2Instruct", "k2_instruct", "k2-v2-instruct", "k2-instruct")),
    )
    for dst_key, model_key, aliases in provider_aliases:
        src_cfg = None
        if isinstance(model_providers, dict):
            src_cfg = model_providers.get(model_key)
        if not isinstance(src_cfg, dict):
            src_cfg = next((providers.get(alias) for alias in aliases if isinstance(providers.get(alias), dict)), None)
        if not isinstance(src_cfg, dict):
            continue
        dst_cfg = providers.setdefault(dst_key, {})
        if src_cfg.get("apiKey") and not dst_cfg.get("apiKey"):
            dst_cfg["apiKey"] = src_cfg.get("apiKey")
        if src_cfg.get("baseUrl") and not dst_cfg.get("apiBase"):
            dst_cfg["apiBase"] = src_cfg.get("baseUrl")

    # Enforce /v1 suffix for K2 API base.
    for provider_key in ("k2Think", "k2Instruct"):
        k2_cfg = providers.get(provider_key)
        if not isinstance(k2_cfg, dict):
            continue
        api_base = k2_cfg.get("apiBase") or k2_cfg.get("api_base")
        if isinstance(api_base, str) and api_base.strip():
            normalized = api_base.rstrip("/")
            if not normalized.endswith("/v1"):
                normalized = f"{normalized}/v1"
            k2_cfg["apiBase"] = normalized
        elif "apiBase" not in k2_cfg and "api_base" not in k2_cfg:
            k2_cfg["apiBase"] = "https://build-api.k2think.ai/v1"

    # Migrate model object format:
    # agents.defaults.model.primary + fallbacks[] -> model + fallbackModel.
    agents = data.get("agents")
    if not isinstance(agents, dict):
        agents = {}
        data["agents"] = agents
    defaults = agents.setdefault("defaults", {})
    model_cfg = defaults.get("model") if isinstance(defaults, dict) else None
    if isinstance(model_cfg, dict):
        primary = model_cfg.get("primary")
        fallbacks = model_cfg.get("fallbacks")
        if isinstance(primary, str) and primary:
            defaults["model"] = primary
        if isinstance(fallbacks, list):
            first = next((item for item in fallbacks if isinstance(item, str) and _is_k2_chat_model(item)), None)
            if first:
                defaults["fallbackModel"] = first

    # K2 defaults for chat models.
    if isinstance(defaults, dict):
        fallback_default = "k2-v2-instruct/LLM360/K2-V2-Instruct"
        defaults.setdefault("model", "k2-think-v2/LLM360/K2-Think-V2")
        defaults.setdefault("fallbackModel", fallback_default)
        model = str(defaults.get("model", ""))
        if not _is_k2_chat_model(model):
            defaults["model"] = "k2-think-v2/LLM360/K2-Think-V2"
        fallback = str(defaults.get("fallbackModel", ""))
        if not _is_k2_chat_model(fallback):
            defaults["fallbackModel"] = fallback_default

    return data


def _is_k2_chat_model(model: str) -> bool:
    m = model.lower()
    return any(
        token in m
        for token in (
            "k2-think",
            "llm360/k2-think",
            "k2-v2-instruct",
            "k2_instruct",
            "k2-instruct",
            "llm360/k2-v2-instruct",
            "llm360/k2-v2",
        )
    )


_PASSTHROUGH_VALUE_MAP_KEYS = {"extra_headers", "env"}
_PASSTHROUGH_CONTAINER_KEYS = {"mcp_servers"}


def convert_keys(data: Any, parent_key: str | None = None) -> Any:
    if isinstance(data, dict):
        # Preserve user-provided keys for header/env maps.
        if parent_key in _PASSTHROUGH_VALUE_MAP_KEYS:
            return {k: convert_keys(v, parent_key=k) for k, v in data.items()}

        # Preserve dynamic map keys (e.g. MCP server names), but still normalize
        # each mapped config object.
        if parent_key in _PASSTHROUGH_CONTAINER_KEYS:
            return {k: convert_keys(v, parent_key=k) for k, v in data.items()}

        converted: dict[str, Any] = {}
        for key, value in data.items():
            normalized_key = camel_to_snake(key)
            converted[normalized_key] = convert_keys(value, parent_key=normalized_key)
        return converted
    if isinstance(data, list):
        return [convert_keys(item, parent_key=parent_key) for item in data]
    return data


def convert_to_camel(data: Any, parent_key: str | None = None) -> Any:
    if isinstance(data, dict):
        if parent_key in _PASSTHROUGH_VALUE_MAP_KEYS:
            return {k: convert_to_camel(v, parent_key=k) for k, v in data.items()}

        if parent_key in _PASSTHROUGH_CONTAINER_KEYS:
            return {k: convert_to_camel(v, parent_key=k) for k, v in data.items()}

        converted: dict[str, Any] = {}
        for key, value in data.items():
            converted[snake_to_camel(key)] = convert_to_camel(value, parent_key=key)
        return converted
    if isinstance(data, list):
        return [convert_to_camel(item, parent_key=parent_key) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    name = name.replace("-", "_")
    if not name:
        return name
    if name.upper() == name:
        return name.lower()
    if re.fullmatch(r"[a-z0-9_]+", name):
        return name

    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.lower()


def snake_to_camel(name: str) -> str:
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """
    Apply K2DO_* environment overrides on top of file data.

    This preserves BaseSettings-like behavior where env vars override config
    file values, while avoiding key mangling for dynamic maps.
    """
    merged = copy.deepcopy(data)
    overrides = _extract_env_overrides()
    _deep_merge_dicts(merged, overrides)
    return merged


def _extract_env_overrides(prefix: str = "K2DO_") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        tail = env_key[len(prefix):]
        if not tail:
            continue

        raw_parts = [p for p in tail.split("__") if p]
        if not raw_parts:
            continue

        path = _normalize_env_path(raw_parts)
        _set_nested_value(overrides, path, _parse_env_value(raw_value))
    return overrides


def _normalize_env_path(raw_parts: list[str]) -> list[str]:
    """
    Normalize env path segments while preserving dynamic map keys.

    Examples:
    - K2DO_TOOLS__MCP_SERVERS__srv1__COMMAND -> tools.mcp_servers.srv1.command
    - K2DO_TOOLS__MCP_SERVERS__srv1__ENV__OPENAI_API_KEY keeps OPENAI_API_KEY
    """
    normalized: list[str] = []
    for part in raw_parts:
        parent = normalized[-1] if normalized else None
        if parent in _PASSTHROUGH_VALUE_MAP_KEYS or parent in _PASSTHROUGH_CONTAINER_KEYS:
            normalized.append(part)
        else:
            normalized.append(_normalize_env_segment(part))
    return normalized


def _normalize_env_segment(segment: str) -> str:
    segment = segment.strip()
    if not segment:
        return segment
    if segment.upper() == segment:
        return segment.lower()
    return camel_to_snake(segment)


def _parse_env_value(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return value
    try:
        return json.loads(raw)
    except Exception:
        return value


def _set_nested_value(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
