from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and handle common aliases/typos."""
    v = (value or "").strip().lower().replace("_", "-").replace(" ", "")
    if v in {"openai", "chatgpt", "gpt"}:
        return "openai"
    elif v in {"anthropic", "claude", "anthorpic"}:
        return "anthropic"
    elif v in {"gemini", "google", "google-genai", "google-generative-ai"}:
        return "gemini"
    elif v in {"ollama"}:
        return "ollama"
    elif v in {"openrouter", "open-router"}:
        return "openrouter"
    elif v in {"custom", "openai-compatible", "local", "vllm"}:
        return "custom"
    return v


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": config.api_key or "EMPTY",
            "base_url": config.base_url or "http://localhost:8000/v1",
        }
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url or "http://localhost:11434",
        }
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenRouter(**kwargs)
        except Exception:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
                "api_key": config.api_key or "",
                "base_url": config.base_url or "https://openrouter.ai/api/v1",
            }
            return ChatOpenAI(**kwargs)

    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
