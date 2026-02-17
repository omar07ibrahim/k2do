"""
Provider Registry — K2DO edition.

LLM chat is intentionally K2-only.
Groq is kept for transcription use-cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """One provider's metadata."""

    name: str
    keywords: tuple[str, ...]
    env_key: str
    display_name: str = ""

    litellm_prefix: str = ""
    skip_prefixes: tuple[str, ...] = ()
    env_extras: tuple[tuple[str, str], ...] = ()

    is_gateway: bool = False
    is_local: bool = False
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    default_api_base: str = ""

    strip_model_prefix: bool = False
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()
    is_oauth: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="k2_think",
        keywords=(
            "k2-think-v2/",
            "k2-think",
            "llm360/k2-think-v2",
            "llm360/k2-think",
        ),
        env_key="OPENAI_API_KEY",
        display_name="K2 Think",
        litellm_prefix="openai",
        skip_prefixes=("openai/",),
        env_extras=(("OPENAI_API_BASE", "{api_base}"),),
        default_api_base="https://build-api.k2think.ai/v1",
    ),
    ProviderSpec(
        name="k2_instruct",
        keywords=(
            "k2-v2-instruct/",
            "k2-v2-instruct",
            "k2_instruct",
            "k2-instruct",
            "llm360/k2-v2-instruct",
            "llm360/k2-v2",
        ),
        env_key="OPENAI_API_KEY",
        display_name="K2 Instruct",
        litellm_prefix="openai",
        skip_prefixes=("openai/",),
        env_extras=(("OPENAI_API_BASE", "{api_base}"),),
        default_api_base="https://build-api.k2think.ai/v1",
    ),
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq (Whisper)",
        litellm_prefix="groq",
        skip_prefixes=("groq/",),
    ),
)


def find_by_model(model: str) -> ProviderSpec | None:
    """Return provider metadata for a supported model identifier."""
    model_lower = model.lower()
    for spec in PROVIDERS:
        if spec.name == "groq":
            continue
        if any(kw in model_lower for kw in spec.keywords):
            return spec

    # Allow bare model names like "LLM360/K2-Think-V2" by defaulting to K2.
    if "llm360/k2-v2-instruct" in model_lower:
        return find_by_name("k2_instruct")
    if "llm360/" in model_lower or "k2-think" in model_lower:
        return find_by_name("k2_think")

    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """K2DO has no LLM gateway auto-detection in K2-only mode."""
    _ = provider_name, api_key, api_base
    return None


def find_by_name(name: str) -> ProviderSpec | None:
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
