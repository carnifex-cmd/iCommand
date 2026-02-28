"""LLM providers for icommand.

Abstract base class for future conversational search support.
All providers are stubs — implementation coming when conversational search is built.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Generate a completion given system and user prompts.

        Args:
            system: System prompt setting the LLM's behavior.
            user: User message / query.

        Returns:
            The LLM's response as a string.
        """
        ...


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider — will call the OpenAI chat completions API."""

    def complete(self, system: str, user: str) -> str:
        # Will use openai.ChatCompletion.create() with the configured model
        raise NotImplementedError("OpenAI LLM provider not yet implemented")


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider — will call a local Ollama instance for inference."""

    def complete(self, system: str, user: str) -> str:
        # Will POST to http://localhost:11434/api/generate with the configured model
        raise NotImplementedError("Ollama LLM provider not yet implemented")


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider — will call the Anthropic messages API."""

    def complete(self, system: str, user: str) -> str:
        # Will use anthropic.Anthropic().messages.create() with the configured model
        raise NotImplementedError("Anthropic LLM provider not yet implemented")


# --- Provider factory ---

_LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAILLMProvider,
    "ollama": OllamaLLMProvider,
    "anthropic": AnthropicLLMProvider,
}


def get_llm_provider(name: str) -> LLMProvider:
    """Get an LLM provider instance by name.

    Args:
        name: Provider name — one of: openai, ollama, anthropic.

    Returns:
        An instance of the requested LLMProvider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    provider_class = _LLM_PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(_LLM_PROVIDERS.keys())
        raise ValueError(f"Unknown LLM provider '{name}'. Available: {available}")
    return provider_class()
