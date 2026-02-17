"""Configuration schema for K2DO."""

from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


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
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""


class ToolsConfig(BaseModel):
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


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
