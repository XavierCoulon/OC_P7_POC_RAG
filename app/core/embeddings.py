"""Embedding providers abstraction for multiple embedding models."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embeddings(self):
        """Get embedding model instance.

        Returns:
            Embedding model instance (MistralAIEmbeddings or HuggingFaceEmbeddings)
        """
        pass

    @abstractmethod
    def get_distance_strategy(self) -> str:
        """Get the distance strategy for this provider.

        Returns:
            Distance strategy string ("COSINE", "EUCLIDEAN_DISTANCE", etc.)
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider.

        Returns:
            Provider name (e.g., "mistral", "huggingface")
        """
        pass


class MistralEmbeddingProvider(EmbeddingProvider):
    """Mistral AI embedding provider."""

    def __init__(self, api_key: str):
        """Initialize Mistral embedding provider.

        Args:
            api_key: Mistral API key
        """
        self.api_key = api_key
        self._embeddings: Optional[MistralAIEmbeddings] = None

    def get_embeddings(self) -> MistralAIEmbeddings:
        """Get Mistral embeddings instance (lazy initialization).

        Returns:
            MistralAIEmbeddings instance
        """
        if self._embeddings is None:
            logger.debug("Initializing MistralAIEmbeddings...")
            self._embeddings = MistralAIEmbeddings(
                model="mistral-embed", api_key=SecretStr(self.api_key)
            )
        return self._embeddings

    def get_distance_strategy(self) -> str:
        """Mistral uses COSINE distance."""
        return "COSINE"

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "mistral"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider (local embeddings)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize HuggingFace embedding provider.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2 for speed/quality balance)
        """
        self.model_name = model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get HuggingFace embeddings instance (lazy initialization).

        Returns:
            HuggingFaceEmbeddings instance
        """
        if self._embeddings is None:
            logger.debug(
                f"Initializing HuggingFaceEmbeddings with model: {self.model_name}..."
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": True},  # For COSINE distance
            )
        return self._embeddings

    def get_distance_strategy(self) -> str:
        """HuggingFace uses COSINE distance (with normalized embeddings)."""
        return "COSINE"

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "huggingface"


def create_embedding_provider(
    provider_name: str, api_key: Optional[str] = None, model_name: Optional[str] = None
) -> EmbeddingProvider:
    """Factory function to create embedding provider.

    Args:
        provider_name: "mistral" or "huggingface"
        api_key: Mistral API key (required if provider_name == "mistral")
        model_name: HuggingFace model name (optional, uses default if not provided)

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider_name is invalid or required parameters are missing
    """
    if provider_name.lower() == "mistral":
        if not api_key:
            raise ValueError("api_key is required for Mistral provider")
        return MistralEmbeddingProvider(api_key)

    elif provider_name.lower() == "huggingface":
        return HuggingFaceEmbeddingProvider(model_name or "all-MiniLM-L6-v2")

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. Choose from: 'mistral', 'huggingface'"
        )
