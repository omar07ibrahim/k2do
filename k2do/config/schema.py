"""Configuration schema for K2DO."""

import unicodedata
from ipaddress import ip_address
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TelegramConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None


class ChannelsConfig(BaseModel):
    """Supported channels."""
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


# ---------------------------------------------------------------------------
# DeepThink: Multi-Agent Parallel Reasoning
# ---------------------------------------------------------------------------

class DeepThinkAgentConfig(BaseModel):
    """One parallel agent in DeepThink mode."""
    name: str = ""
    model: str = ""
    temperature: float = 0.7
    role_prompt: str = ""


class DeepThinkConfig(BaseModel):
    """Multi-agent DeepThink configuration."""
    enabled: bool = True
    complexity_threshold: float = 0.45
    max_agents: int = 3
    judge_model: str = ""
    thinker_timeout_s: float = 90.0
    judge_timeout_s: float = 60.0
    agents: list[DeepThinkAgentConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent / Provider
# ---------------------------------------------------------------------------

class AgentDefaults(BaseModel):
    workspace: str = "~/.k2do/workspace"
    model: str = "k2-think-v2/LLM360/K2-Think-V2"
    fallback_model: str = "k2-v2-instruct/LLM360/K2-V2-Instruct"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    memory_window: int = 50
    context_max_chars: int = 24000
    simple_route_max_complexity: float = 0.2
    store_reasoning: bool = False


class AgentsConfig(BaseModel):
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    deepthink: DeepThinkConfig = Field(default_factory=DeepThinkConfig)


class ProviderConfig(BaseModel):
    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None


class K2ThinkProviderConfig(ProviderConfig):
    api_base: str | None = "https://build-api.k2think.ai/v1"


class K2InstructProviderConfig(ProviderConfig):
    api_base: str | None = "https://build-api.k2think.ai/v1"


class ProvidersConfig(BaseModel):
    k2_think: K2ThinkProviderConfig = Field(default_factory=K2ThinkProviderConfig)
    k2_instruct: K2InstructProviderConfig = Field(default_factory=K2InstructProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchConfig(BaseModel):
    api_key: str = ""
    max_results: int = 5


class WebToolsConfig(BaseModel):
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    timeout: int = 60


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(default="", max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=32)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = Field(default="", max_length=4096)
    protocol_mode: Literal["auto", "legacy"] = "auto"
    connect_timeout_seconds: float = Field(default=10.0, ge=0.1, le=120.0)
    call_timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0)
    max_output_bytes: int = Field(default=16 * 1024, ge=256, le=64 * 1024)

    @field_validator("command", "url")
    @classmethod
    def _validate_transport_text(cls, value: str) -> str:
        if any(
            ord(character) == 127
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in value
        ):
            raise ValueError("MCP transport value contains control characters")
        return value

    @field_validator("args")
    @classmethod
    def _validate_args(cls, values: list[str]) -> list[str]:
        if any(
            type(value) is not str
            or len(value.encode("utf-8")) > 4096
            or "\x00" in value
            for value in values
        ):
            raise ValueError("MCP arguments are invalid")
        return values

    @field_validator("env")
    @classmethod
    def _validate_env(cls, values: dict[str, str]) -> dict[str, str]:
        if len(values) > 64:
            raise ValueError("MCP environment has too many entries")
        for key, value in values.items():
            if (
                not key
                or len(key) > 128
                or not (key[0].isalpha() or key[0] == "_")
                or any(not (character.isalnum() or character == "_") for character in key)
                or len(value.encode("utf-8")) > 16 * 1024
                or "\x00" in value
            ):
                raise ValueError("MCP environment contains an invalid entry")
        return values

    @model_validator(mode="after")
    def _validate_transport(self) -> Self:
        if bool(self.command) == bool(self.url):
            raise ValueError("MCP server must configure exactly one transport")
        if self.url:
            if any(ord(character) < 33 or ord(character) > 126 for character in self.url):
                raise ValueError("MCP URL must contain printable ASCII characters only")
            if "\\" in self.url:
                raise ValueError("MCP URL must not contain backslashes")
            if "#" in self.url:
                raise ValueError("MCP URL must not include a fragment")
            # MCP endpoints in K2DO do not use URL query configuration. Refusing
            # every query avoids putting credentials in URLs and keeps the URL's
            # meaning identical across urllib and the SDK's httpx2 transport.
            if "?" in self.url:
                raise ValueError("MCP URL must not include a query string")

            try:
                parsed = urlsplit(self.url)
                host = parsed.hostname
                # Accessing ``port`` also rejects malformed and out-of-range ports.
                port = parsed.port
            except ValueError as error:
                raise ValueError("MCP URL is invalid") from error

            if parsed.scheme not in {"http", "https"} or not host:
                raise ValueError("MCP URL must use HTTP or HTTPS and include a host")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("MCP URL must not include user information")
            if "%" in parsed.netloc:
                raise ValueError("MCP URL authority must not use percent encoding")
            if parsed.netloc.endswith(":") or port == 0:
                raise ValueError("MCP URL port is invalid")

            if parsed.scheme == "http" and not _is_local_mcp_host(host):
                raise ValueError("Remote MCP URLs must use HTTPS")
        return self


def _is_local_mcp_host(host: str) -> bool:
    """Return whether cleartext HTTP is safe for this explicit local host."""
    if host.casefold() == "localhost":
        return True
    if "%" in host:
        return False
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class ToolsConfig(BaseModel):
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @field_validator("mcp_servers")
    @classmethod
    def _validate_server_names(
        cls,
        values: dict[str, MCPServerConfig],
    ) -> dict[str, MCPServerConfig]:
        if len(values) > 16:
            raise ValueError("Too many MCP servers are configured")
        if any(
            not name
            or len(name.encode("utf-8")) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            for name in values
        ):
            raise ValueError("MCP server name is invalid")
        return values


class Config(BaseModel):
    """Root configuration for K2DO."""
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        model_lower = (model or self.agents.defaults.model).lower()
        if any(x in model_lower for x in ("k2-v2-instruct", "k2_instruct", "k2-instruct")):
            return self.providers.k2_instruct, "k2_instruct"
        return self.providers.k2_think, "k2_think"

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        if name in {"k2_think", "k2_instruct"}:
            return "https://build-api.k2think.ai/v1"
        return None

    model_config = ConfigDict(
        extra="ignore",
    )
