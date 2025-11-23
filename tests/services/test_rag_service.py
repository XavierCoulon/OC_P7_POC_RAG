"""Tests for RAG Service."""

from unittest.mock import Mock, patch

import pytest

from app.core.classification import INTENT_CHAT, INTENT_RAG
from app.services.rag_service import RAGService


@pytest.fixture
def rag_service():
    """Create RAG service instance for testing."""
    service = RAGService()
    return service


class TestRAGServiceInitialization:
    """Test RAG service initialization."""

    def test_service_initializes_correctly(self, rag_service):
        """Test that RAG service initializes with correct state."""
        assert rag_service is not None
        assert rag_service.RETRIEVER_K == 6
        assert rag_service.DISTANCE_STRATEGY == "COSINE"
        assert rag_service.SUPPORTED_PROVIDERS == ["mistral", "huggingface"]

    def test_providers_dict_initialized(self, rag_service):
        """Test that provider dictionaries are properly initialized."""
        for provider in rag_service.SUPPORTED_PROVIDERS:
            assert provider in rag_service.vector_stores
            assert provider in rag_service.rag_chains
            assert provider in rag_service.index_managers
            assert rag_service.vector_stores[provider] is None
            assert rag_service.rag_chains[provider] is None
            assert rag_service.index_managers[provider] is None


class TestProviderHandling:
    """Test provider validation and handling."""

    def test_unsupported_provider_raises_error(self, rag_service):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            rag_service._get_or_create_embedding_provider("invalid_provider")

    def test_supported_providers_list(self, rag_service):
        """Test that supported providers can be retrieved."""
        assert "mistral" in rag_service.SUPPORTED_PROVIDERS
        assert "huggingface" in rag_service.SUPPORTED_PROVIDERS

    def test_is_ready_returns_false_initially(self, rag_service):
        """Test that service reports not ready before loading."""
        assert rag_service.is_ready("mistral") is False
        assert rag_service.is_ready("huggingface") is False

    def test_is_ready_with_invalid_provider(self, rag_service):
        """Test is_ready with invalid provider returns False."""
        assert rag_service.is_ready("invalid") is False


class TestEventExtraction:
    """Test event extraction from context documents."""

    def test_extract_events_from_empty_context(self, rag_service):
        """Test extracting events from empty context."""
        events = rag_service._extract_events_from_context([])
        assert events == []

    def test_extract_events_from_context(self, rag_service):
        """Test extracting event information from documents."""
        # Mock document with metadata
        mock_doc = Mock()
        mock_doc.metadata = {
            "originagenda_title": "Test Event",
            "location_city": "Pau",
            "location_postalcode": "64000",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com/event",
        }

        events = rag_service._extract_events_from_context([mock_doc])

        assert len(events) == 1
        assert events[0]["title"] == "Test Event"
        assert events[0]["location"] == "Pau (64000)"
        assert events[0]["start_date"] == "2025-06-21"
        assert events[0]["url"] == "https://example.com/event"

    def test_extract_events_with_missing_metadata(self, rag_service):
        """Test extracting events with missing metadata fields."""
        mock_doc = Mock()
        mock_doc.metadata = {}

        events = rag_service._extract_events_from_context([mock_doc])

        assert len(events) == 1
        assert events[0]["title"] == "Événement sans titre"
        assert events[0]["location"] == "Lieu non spécifié (CP)"
        assert events[0]["start_date"] == "Date non spécifiée"
        assert events[0]["url"] is None

    def test_extract_events_removes_duplicates(self, rag_service):
        """Test that duplicate event titles are removed."""
        mock_doc1 = Mock()
        mock_doc1.metadata = {
            "originagenda_title": "Same Event",
            "location_city": "Pau",
            "location_postalcode": "64000",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com/event1",
        }

        mock_doc2 = Mock()
        mock_doc2.metadata = {
            "originagenda_title": "Same Event",
            "location_city": "Bayonne",
            "location_postalcode": "64100",
            "firstdate_begin": "2025-06-22",
            "canonicalurl": "https://example.com/event2",
        }

        events = rag_service._extract_events_from_context([mock_doc1, mock_doc2])

        # Should only have 1 event (duplicate removed)
        assert len(events) == 1
        assert events[0]["title"] == "Same Event"


class TestIntentClassification:
    """Test query intent classification."""

    @patch("app.services.rag_service.classify_query_intent")
    def test_classify_intent_rag(self, mock_classify, rag_service):
        """Test RAG intent classification."""
        mock_classify.return_value = INTENT_RAG

        intent = rag_service.classify_intent("Quels événements en juin ?")

        assert intent == INTENT_RAG
        mock_classify.assert_called_once()

    @patch("app.services.rag_service.classify_query_intent")
    def test_classify_intent_chat(self, mock_classify, rag_service):
        """Test CHAT intent classification."""
        mock_classify.return_value = INTENT_CHAT

        intent = rag_service.classify_intent("Bonjour !")

        assert intent == INTENT_CHAT
        mock_classify.assert_called_once()


class TestAnswerQuestion:
    """Test answer_question method."""

    @patch("app.services.rag_service.RAGService.classify_intent")
    @patch("app.services.rag_service.get_chat_response")
    def test_answer_question_chat_intent(self, mock_chat_response, mock_classify, rag_service):
        """Test answering a question with CHAT intent."""
        mock_classify.return_value = INTENT_CHAT
        mock_chat_response.return_value = "Bonjour! Comment puis-je vous aider?"

        result = rag_service.answer_question("Bonjour")

        assert result["status"] == "success"
        assert result["intent"] == INTENT_CHAT
        assert result["answer"] == "Bonjour! Comment puis-je vous aider?"
        assert result["events"] == []

    def test_answer_question_with_invalid_provider(self, rag_service):
        """Test answering with invalid provider."""
        result = rag_service.answer_question("Test", provider="invalid_provider")

        assert result["status"] == "error"
        assert "Unsupported provider" in result["answer"]

    @patch("app.services.rag_service.RAGService.classify_intent")
    def test_answer_question_rag_not_ready(self, mock_classify, rag_service):
        """Test RAG when index is not ready."""
        mock_classify.return_value = INTENT_RAG

        result = rag_service.answer_question("Quels événements ?", provider="mistral")

        assert result["status"] == "success"
        assert "Je n'ai pas accès à l'index" in result["answer"]
        assert result["intent"] == INTENT_RAG
        assert result["events"] == []

    @patch("app.services.rag_service.RAGService.classify_intent")
    @patch("app.services.rag_service.RAGService.get_rag_chain")
    @patch("app.services.rag_service.RAGService._invoke_with_retry")
    def test_answer_question_geographic_validation_blocks_events(
        self, mock_invoke, mock_get_chain, mock_classify, rag_service
    ):
        """Test that geographic validation blocks events from being returned."""
        mock_classify.return_value = INTENT_RAG
        mock_chain = Mock()
        mock_get_chain.return_value = mock_chain
        rag_service.rag_chains["mistral"] = mock_chain

        # Mock RAG response with geographic validation message
        mock_invoke.return_value = {
            "answer": "Je suis spécialisé uniquement dans Pyrénées-Atlantiques. "
            "Je ne dispose pas d'événements pour les autres régions.",
            "context": [Mock(metadata={"originagenda_title": "Event"})],
        }

        result = rag_service.answer_question("Des concerts à Paris ?", provider="mistral")

        assert result["status"] == "success"
        assert result["events"] == []
        assert "Je suis spécialisé uniquement dans" in result["answer"]

    @patch("app.services.rag_service.RAGService.classify_intent")
    @patch("app.services.rag_service.RAGService.get_rag_chain")
    @patch("app.services.rag_service.RAGService._invoke_with_retry")
    def test_answer_question_no_style_returns_events(self, mock_invoke, mock_get_chain, mock_classify, rag_service):
        """Test that 'no style' responses still return events."""
        mock_classify.return_value = INTENT_RAG
        mock_chain = Mock()
        mock_get_chain.return_value = mock_chain
        rag_service.rag_chains["mistral"] = mock_chain

        mock_doc = Mock()
        mock_doc.metadata = {
            "originagenda_title": "Concert",
            "location_city": "Anglet",
            "location_postalcode": "64600",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com",
        }

        # Mock RAG response with "no style" message
        mock_invoke.return_value = {
            "answer": "Aucun événement de ce style trouvé dans cette ville, "
            "mais voici d'autres événements disponibles.",
            "context": [mock_doc],
        }

        result = rag_service.answer_question("Des concerts à Anglet ?", provider="mistral")

        assert result["status"] == "success"
        assert len(result["events"]) == 1
        assert result["events"][0]["title"] == "Concert"


class TestIndexInfo:
    """Test index information retrieval."""

    def test_get_index_info_not_initialized(self, rag_service):
        """Test getting index info when not initialized."""
        info = rag_service.get_index_info("mistral")

        assert info["status"] == "not_initialized"
        assert info["provider"] == "mistral"


class TestAvailableProviders:
    """Test provider availability checking."""

    def test_get_available_providers(self, rag_service):
        """Test getting available providers status."""
        providers = rag_service.get_available_providers()

        assert "mistral" in providers
        assert "huggingface" in providers
        assert providers["mistral"]["available"] is False
        assert providers["huggingface"]["available"] is False
