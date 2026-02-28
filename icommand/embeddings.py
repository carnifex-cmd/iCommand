"""Embedding providers for icommand.

Abstract base class with a local ONNX-based implementation using all-MiniLM-L6-v2.
No PyTorch required — uses onnxruntime for Python 3.9–3.13+ compatibility.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

# Suppress HuggingFace Hub's unauthenticated request warning — the model is
# downloaded once and cached locally; no token is needed for public models.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


# ONNX model identifier on Hugging Face Hub
_ONNX_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_MODEL_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"


class EmbeddingProvider(ABC):
    """Base class for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...


class LocalProvider(EmbeddingProvider):
    """Local embedding provider using all-MiniLM-L6-v2 via ONNX runtime.

    - No PyTorch required
    - Model downloads on first use (~90MB) via huggingface-hub
    - Compatible with Python 3.9–3.13+
    """

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the ONNX model and tokenizer on first use."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(
            repo_id=_ONNX_MODEL_REPO,
            filename=_ONNX_MODEL_FILE,
        )
        tokenizer_path = hf_hub_download(
            repo_id=_ONNX_MODEL_REPO,
            filename=_TOKENIZER_FILE,
        )

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=128
        )
        self._tokenizer.enable_truncation(max_length=128)

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Apply mean pooling to token embeddings."""
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the local ONNX model."""
        if self._session is None:
            self._load()

        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # outputs[0] = last hidden state: (batch, seq_len, hidden_size)
        token_embeddings = outputs[0]
        pooled = self._mean_pool(token_embeddings, attention_mask)
        normalized = self._normalize(pooled)

        return normalized.tolist()


# --- Stub providers for future implementation ---


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider — will use text-embedding-3-small."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAI embedding provider not yet implemented")


class AnthropicProvider(EmbeddingProvider):
    """Anthropic embedding provider — will use Voyage AI embeddings."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic embedding provider not yet implemented")


class OllamaProvider(EmbeddingProvider):
    """Ollama embedding provider — will use local Ollama instance."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Ollama embedding provider not yet implemented")


# --- Provider factory ---

_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "local": LocalProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str) -> EmbeddingProvider:
    """Get an embedding provider instance by name."""
    provider_class = _PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Unknown embedding provider '{name}'. Available: {available}")
    return provider_class()
