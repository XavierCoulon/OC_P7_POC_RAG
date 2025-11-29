"""Tests for embedding providers."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.core.embeddings import (
    EmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    MistralEmbeddingProvider,
    create_embedding_provider,
)


class TestMistralEmbeddingProvider:
    """Test Mistral embedding provider."""

    def test_mistral_initialization(self):
        """Test Mistral provider initializes correctly."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider._embeddings is None  # Lazy initialization

    def test_mistral_get_distance_strategy(self):
        """Test Mistral uses COSINE distance."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.get_distance_strategy() == "COSINE"

    def test_mistral_get_provider_name(self):
        """Test Mistral provider returns correct name."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        assert provider.get_provider_name() == "mistral"

    @patch("app.core.embeddings.MistralAIEmbeddings")
    def test_mistral_get_embeddings_lazy_initialization(self, mock_mistral):
        """Test Mistral embeddings are lazy initialized on first call."""
        mock_embeddings_instance = MagicMock()
        mock_mistral.return_value = mock_embeddings_instance

        provider = MistralEmbeddingProvider(api_key="test-key")

        # First call - should initialize
        embeddings1 = provider.get_embeddings()
        assert embeddings1 == mock_embeddings_instance
        assert mock_mistral.call_count == 1

        # Second call - should return cached instance
        embeddings2 = provider.get_embeddings()
        assert embeddings2 == mock_embeddings_instance
        assert mock_mistral.call_count == 1  # No additional call

    @patch("app.core.embeddings.MistralAIEmbeddings")
    def test_mistral_get_embeddings_uses_correct_model(self, mock_mistral):
        """Test Mistral embeddings uses mistral-embed model."""
        provider = MistralEmbeddingProvider(api_key="test-key")
        provider.get_embeddings()

        # Verify called with correct model
        mock_mistral.assert_called_once()
        call_kwargs = mock_mistral.call_args[1]
        assert call_kwargs["model"] == "mistral-embed"

    @patch("app.core.embeddings.MistralAIEmbeddings")
    def test_mistral_logging_on_initialization(self, mock_mistral, caplog):
        """Test that debug log is created on first initialization."""
        provider = MistralEmbeddingProvider(api_key="test-key")

        with caplog.at_level(logging.DEBUG):
            provider.get_embeddings()

        assert "Initializing MistralAIEmbeddings" in caplog.text


class TestHuggingFaceEmbeddingProvider:
    """Test HuggingFace embedding provider."""

    def test_huggingface_initialization_default_model(self):
        """Test HuggingFace provider initializes with default model."""
        provider = HuggingFaceEmbeddingProvider()
        assert provider.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert provider._embeddings is None  # Lazy initialization

    def test_huggingface_initialization_custom_model(self):
        """Test HuggingFace provider initializes with custom model."""
        provider = HuggingFaceEmbeddingProvider(model_name="custom-model")
        assert provider.model_name == "custom-model"

    def test_huggingface_get_distance_strategy(self):
        """Test HuggingFace uses COSINE distance."""
        provider = HuggingFaceEmbeddingProvider()
        assert provider.get_distance_strategy() == "COSINE"

    def test_huggingface_get_provider_name(self):
        """Test HuggingFace provider returns correct name."""
        provider = HuggingFaceEmbeddingProvider()
        assert provider.get_provider_name() == "huggingface"

    @patch("app.core.embeddings.HuggingFaceEmbeddings")
    def test_huggingface_get_embeddings_lazy_initialization(self, mock_hf):
        """Test HuggingFace embeddings are lazy initialized on first call."""
        mock_embeddings_instance = MagicMock()
        mock_hf.return_value = mock_embeddings_instance

        provider = HuggingFaceEmbeddingProvider()

        # First call - should initialize
        embeddings1 = provider.get_embeddings()
        assert embeddings1 == mock_embeddings_instance
        assert mock_hf.call_count == 1

        # Second call - should return cached instance
        embeddings2 = provider.get_embeddings()
        assert embeddings2 == mock_embeddings_instance
        assert mock_hf.call_count == 1  # No additional call

    @patch("app.core.embeddings.HuggingFaceEmbeddings")
    def test_huggingface_get_embeddings_with_default_model(self, mock_hf):
        """Test HuggingFace embeddings uses default multilingual model."""
        provider = HuggingFaceEmbeddingProvider()
        provider.get_embeddings()

        # Verify called with correct model and normalize_embeddings
        mock_hf.assert_called_once()
        call_kwargs = mock_hf.call_args[1]
        assert call_kwargs["model_name"] == "paraphrase-multilingual-MiniLM-L12-v2"
        assert call_kwargs["encode_kwargs"]["normalize_embeddings"] is True

    @patch("app.core.embeddings.HuggingFaceEmbeddings")
    def test_huggingface_get_embeddings_with_custom_model(self, mock_hf):
        """Test HuggingFace embeddings uses custom model."""
        provider = HuggingFaceEmbeddingProvider(model_name="another-model")
        provider.get_embeddings()

        call_kwargs = mock_hf.call_args[1]
        assert call_kwargs["model_name"] == "another-model"

    @patch("app.core.embeddings.HuggingFaceEmbeddings")
    def test_huggingface_logging_on_initialization(self, mock_hf, caplog):
        """Test that debug log includes model name on initialization."""
        provider = HuggingFaceEmbeddingProvider(model_name="test-model")

        with caplog.at_level(logging.DEBUG):
            provider.get_embeddings()

        assert "Initializing HuggingFaceEmbeddings" in caplog.text
        assert "test-model" in caplog.text


class TestEmbeddingProviderAbstract:
    """Test EmbeddingProvider abstract base class."""

    def test_embedding_provider_abstract_methods(self):
        """Test that EmbeddingProvider defines required abstract methods."""
        assert hasattr(EmbeddingProvider, "get_embeddings")
        assert hasattr(EmbeddingProvider, "get_distance_strategy")
        assert hasattr(EmbeddingProvider, "get_provider_name")

    def test_concrete_provider_implements_abstract_methods(self):
        """Test that concrete providers implement all abstract methods."""
        mistral_provider = MistralEmbeddingProvider(api_key="test")
        huggingface_provider = HuggingFaceEmbeddingProvider()

        # All abstract methods should be callable
        assert callable(mistral_provider.get_embeddings)
        assert callable(mistral_provider.get_distance_strategy)
        assert callable(mistral_provider.get_provider_name)

        assert callable(huggingface_provider.get_embeddings)
        assert callable(huggingface_provider.get_distance_strategy)
        assert callable(huggingface_provider.get_provider_name)


class TestCreateEmbeddingProvider:
    """Test factory function for creating embedding providers."""

    @patch("app.core.embeddings.MistralAIEmbeddings")
    def test_create_mistral_provider(self, mock_mistral):
        """Test creating Mistral provider via factory."""
        provider = create_embedding_provider("mistral", api_key="test-key")

        assert isinstance(provider, MistralEmbeddingProvider)
        assert provider.api_key == "test-key"

    def test_create_mistral_provider_case_insensitive(self):
        """Test factory is case-insensitive for provider name."""
        provider1 = create_embedding_provider("MISTRAL", api_key="test-key")
        provider2 = create_embedding_provider("Mistral", api_key="test-key")
        provider3 = create_embedding_provider("mistral", api_key="test-key")

        assert isinstance(provider1, MistralEmbeddingProvider)
        assert isinstance(provider2, MistralEmbeddingProvider)
        assert isinstance(provider3, MistralEmbeddingProvider)

    def test_create_mistral_without_api_key_raises_error(self):
        """Test creating Mistral provider without API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required for Mistral provider"):
            create_embedding_provider("mistral")

    def test_create_mistral_with_none_api_key_raises_error(self):
        """Test creating Mistral provider with None API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required for Mistral provider"):
            create_embedding_provider("mistral", api_key=None)

    def test_create_huggingface_provider_default_model(self):
        """Test creating HuggingFace provider with default model."""
        provider = create_embedding_provider("huggingface")

        assert isinstance(provider, HuggingFaceEmbeddingProvider)
        assert provider.model_name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_create_huggingface_provider_custom_model(self):
        """Test creating HuggingFace provider with custom model."""
        provider = create_embedding_provider("huggingface", model_name="custom-model")

        assert isinstance(provider, HuggingFaceEmbeddingProvider)
        assert provider.model_name == "custom-model"

    def test_create_huggingface_provider_case_insensitive(self):
        """Test factory is case-insensitive for HuggingFace."""
        provider1 = create_embedding_provider("HUGGINGFACE")
        provider2 = create_embedding_provider("HuggingFace")
        provider3 = create_embedding_provider("huggingface")

        assert isinstance(provider1, HuggingFaceEmbeddingProvider)
        assert isinstance(provider2, HuggingFaceEmbeddingProvider)
        assert isinstance(provider3, HuggingFaceEmbeddingProvider)

    def test_create_invalid_provider_raises_error(self):
        """Test creating invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider("invalid-provider")

    def test_create_invalid_provider_shows_valid_options(self):
        """Test error message includes valid provider options."""
        with pytest.raises(ValueError) as exc_info:
            create_embedding_provider("invalid-provider")

        error_message = str(exc_info.value)
        assert "mistral" in error_message.lower()
        assert "huggingface" in error_message.lower()

    def test_create_provider_ignores_api_key_for_huggingface(self):
        """Test that api_key is ignored when creating HuggingFace provider."""
        provider = create_embedding_provider("huggingface", api_key="ignored-key")
        assert isinstance(provider, HuggingFaceEmbeddingProvider)

    def test_create_provider_returns_embedding_provider_instance(self):
        """Test that factory returns EmbeddingProvider instances."""
        mistral = create_embedding_provider("mistral", api_key="test")
        huggingface = create_embedding_provider("huggingface")

        assert isinstance(mistral, EmbeddingProvider)
        assert isinstance(huggingface, EmbeddingProvider)
