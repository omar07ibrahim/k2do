"""LiteLLM provider implementation for multi-provider support."""

import json_repair
import os
from typing import Any

import litellm
from litellm import acompletion

from k2do.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from k2do.providers.registry import find_by_model, find_by_name, find_gateway
from k2do.utils.helpers import sanitize_model_output


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM.

    K2DO chat models are K2 Think + K2 Instruct.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "k2-think-v2/LLM360/K2-Think-V2",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        provider_configs: dict[str, dict[str, Any]] | None = None,
        request_timeout: float | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.provider_configs = provider_configs or {}
        # Avoid indefinite hangs on provider/network side.
        self.request_timeout = (
            float(request_timeout)
            if request_timeout is not None
            else float(os.getenv("K2DO_LLM_TIMEOUT", "90"))
        )
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model, provider_name=provider_name)
        for name, cfg in self.provider_configs.items():
            key = cfg.get("api_key")
            if not key:
                continue
            self._setup_env(
                api_key=key,
                api_base=cfg.get("api_base"),
                model=cfg.get("model", default_model),
                provider_name=name,
            )
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for provider backends.
        litellm.drop_params = True
    
    def _setup_env(
        self,
        api_key: str,
        api_base: str | None,
        model: str,
        provider_name: str | None = None,
    ) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or (find_by_name(provider_name) if provider_name else None) or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # Provider spec has no env-key based auth.
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            if spec.name in {"k2_think", "k2_instruct"}:
                os.environ[spec.env_key] = api_key
            else:
                os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            if spec.name in {"k2_think", "k2_instruct"}:
                os.environ[env_name] = resolved
            else:
                os.environ.setdefault(env_name, resolved)

    def _resolve_request_config(self, model: str) -> tuple[str | None, str | None, dict[str, str]]:
        """Resolve API credentials and headers for the requested model."""
        spec = find_by_model(model)
        api_key = self.api_key
        api_base = self.api_base
        headers = dict(self.extra_headers)

        if spec:
            cfg = self.provider_configs.get(spec.name) or {}
            if cfg.get("api_key"):
                api_key = cfg["api_key"]
            if cfg.get("api_base"):
                api_base = cfg["api_base"]
            extra_headers = cfg.get("extra_headers")
            if isinstance(extra_headers, dict):
                headers.update(extra_headers)

        return api_key, api_base, headers
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        model = self._normalize_k2_model_name(model)

        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model

    @staticmethod
    def _normalize_k2_model_name(model: str) -> str:
        """
        Normalize accepted K2 model ids to the raw upstream model id.

        Examples:
        - k2-think-v2/LLM360/K2-Think-V2 -> LLM360/K2-Think-V2
        - k2-v2-instruct/LLM360/K2-V2-Instruct -> LLM360/K2-V2-Instruct
        - k2_think/LLM360/K2-Think-V2    -> LLM360/K2-Think-V2
        """
        if "/" not in model:
            return model
        provider_prefixes = {
            "k2-think-v2",
            "k2_think",
            "k2think",
            "k2-v2-instruct",
            "k2_instruct",
            "k2instruct",
        }
        first, rest = model.split("/", 1)
        if first.lower() in provider_prefixes and rest:
            return rest
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'k2-think-v2/LLM360/K2-Think-V2').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        raw_model = model or self.default_model
        req_api_key, req_api_base, req_headers = self._resolve_request_config(raw_model)
        model = self._resolve_model(raw_model)
        
        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.request_timeout,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if req_api_key:
            kwargs["api_key"] = req_api_key
        
        # Pass api_base for custom endpoints
        if req_api_base:
            kwargs["api_base"] = req_api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if req_headers:
            kwargs["extra_headers"] = req_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise RuntimeError(f"LLM request failed for model '{raw_model}': {e}") from e
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        raw_reasoning = getattr(message, "reasoning_content", None)
        if raw_reasoning is None or isinstance(raw_reasoning, str):
            reasoning_content = sanitize_model_output(raw_reasoning)
        else:
            reasoning_content = str(raw_reasoning)

        raw_content = message.content
        if raw_content is None or isinstance(raw_content, str):
            content = sanitize_model_output(raw_content)
        else:
            content = str(raw_content)
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
