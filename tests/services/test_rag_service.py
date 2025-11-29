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
            "title": "Test Event",
            "uid": "event-123",
            "location_city": "Pau",
            "location_department": "Pyrénées-Atlantiques",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com/event",
        }

        events = rag_service._extract_events_from_context([mock_doc])

        assert len(events) == 1
        assert events[0]["title"] == "Test Event"
        assert events[0]["location"] == "Pau (Pyrénées-Atlantiques)"
        assert events[0]["start_date"] == "2025-06-21"
        assert events[0]["url"] == "https://example.com/event"

    def test_extract_events_with_missing_metadata(self, rag_service):
        """Test extracting events with missing metadata fields."""
        mock_doc = Mock()
        mock_doc.metadata = {}

        events = rag_service._extract_events_from_context([mock_doc])

        assert len(events) == 1
        assert events[0]["title"] == "Titre inconnu"
        # Location is formatted as "city (dept)" but both are empty, so it becomes "()" then stripped
        assert events[0]["location"] == "()"
        assert events[0]["start_date"] is None
        assert events[0]["url"] is None

    def test_extract_events_removes_duplicates(self, rag_service):
        """Test that duplicate event UIDs are removed."""
        mock_doc1 = Mock()
        mock_doc1.metadata = {
            "title": "Same Event",
            "uid": "event-123",  # Same UID
            "location_city": "Pau",
            "location_department": "Pyrénées-Atlantiques",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com/event1",
        }

        mock_doc2 = Mock()
        mock_doc2.metadata = {
            "title": "Same Event",
            "uid": "event-123",  # Same UID - should be filtered
            "location_city": "Bayonne",
            "location_department": "Pyrénées-Atlantiques",
            "firstdate_begin": "2025-06-22",
            "canonicalurl": "https://example.com/event2",
        }

        events = rag_service._extract_events_from_context([mock_doc1, mock_doc2])

        # Should only have 1 event (duplicate UID removed)
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
        """Test that events are always extracted from context (no longer blocked)."""
        mock_classify.return_value = INTENT_RAG
        mock_chain = Mock()
        mock_get_chain.return_value = mock_chain
        rag_service.rag_chains["mistral"] = mock_chain

        # Mock RAG response with geographic validation message
        mock_doc = Mock()
        mock_doc.metadata = {
            "title": "Event",
            "uid": "event-123",
            "location_city": "Paris",
            "location_department": "Île-de-France",
            "firstdate_begin": "2025-06-21",
            "canonicalurl": "https://example.com",
        }

        mock_invoke.return_value = {
            "answer": "Je suis spécialisé uniquement dans Pyrénées-Atlantiques. "
            "Je ne dispose pas d'événements pour les autres régions.",
            "context": [mock_doc],
        }

        result = rag_service.answer_question("Des concerts à Paris ?", provider="mistral")

        assert result["status"] == "success"
        # Events are now ALWAYS extracted for transparency (even if out of scope)
        assert len(result["events"]) == 1
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
            "title": "Concert",
            "uid": "event-123",
            "location_city": "Anglet",
            "location_department": "Pyrénées-Atlantiques",
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


class TestLLMInitialization:
    """Test LLM initialization and caching."""

    @patch("app.services.rag_service.ChatMistralAI")
    def test_initialize_llm_creates_instance(self, mock_llm_class, rag_service):
        """Test that _initialize_llm creates ChatMistralAI instance."""
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance

        result = rag_service._initialize_llm()

        assert result == mock_llm_instance
        mock_llm_class.assert_called_once()

    @patch("app.services.rag_service.ChatMistralAI")
    def test_get_llm_lazy_initialization(self, mock_llm_class, rag_service):
        """Test that _get_llm initializes LLM only once (lazy caching)."""
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance

        # First call should initialize
        llm1 = rag_service._get_llm()
        assert llm1 == mock_llm_instance
        assert mock_llm_class.call_count == 1

        # Second call should return cached instance
        llm2 = rag_service._get_llm()
        assert llm2 == mock_llm_instance
        assert mock_llm_class.call_count == 1  # Still 1, not 2

    @patch("app.services.rag_service.ChatMistralAI")
    def test_initialize_llm_correct_parameters(self, mock_llm_class, rag_service):
        """Test that _initialize_llm passes correct parameters."""
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance

        rag_service._initialize_llm()

        # Verify it was called with correct model and parameters
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model_name"] == "mistral-small-latest"
        assert call_kwargs["temperature"] == 0.3


class TestInvokeWithRetry:
    """Test retry logic with exponential backoff."""

    def test_invoke_with_retry_success_first_attempt(self, rag_service):
        """Test successful invocation on first attempt."""
        mock_invokable = Mock()
        expected_result = {"answer": "Test response"}
        mock_invokable.invoke.return_value = expected_result

        result = rag_service._invoke_with_retry(mock_invokable, {"input": "test"})

        assert result == expected_result
        mock_invokable.invoke.assert_called_once()

    def test_invoke_with_retry_429_error_then_success(self, rag_service):
        """Test retry on 429 error followed by success."""
        mock_invokable = Mock()
        expected_result = {"answer": "Success"}

        # First call raises 429, second succeeds
        mock_invokable.invoke.side_effect = [
            Exception("429: Too Many Requests"),
            expected_result,
        ]

        result = rag_service._invoke_with_retry(mock_invokable, {"input": "test"}, max_retries=3, initial_delay=0.01)

        assert result == expected_result
        assert mock_invokable.invoke.call_count == 2

    def test_invoke_with_retry_capacity_exceeded_then_success(self, rag_service):
        """Test retry on capacity exceeded error."""
        mock_invokable = Mock()
        expected_result = {"answer": "Success"}

        mock_invokable.invoke.side_effect = [
            Exception("capacity exceeded"),
            expected_result,
        ]

        result = rag_service._invoke_with_retry(mock_invokable, {"input": "test"}, max_retries=3, initial_delay=0.01)

        assert result == expected_result
        assert mock_invokable.invoke.call_count == 2

    def test_invoke_with_retry_rate_limit_then_success(self, rag_service):
        """Test retry on rate limit error."""
        mock_invokable = Mock()
        expected_result = {"answer": "Success"}

        mock_invokable.invoke.side_effect = [
            Exception("rate limit exceeded"),
            expected_result,
        ]

        result = rag_service._invoke_with_retry(mock_invokable, {"input": "test"}, max_retries=3, initial_delay=0.01)

        assert result == expected_result

    def test_invoke_with_retry_exhausts_retries_429(self, rag_service):
        """Test that 429 error is returned after max retries exhausted."""
        mock_invokable = Mock()
        mock_invokable.invoke.side_effect = Exception("429: Too Many Requests")

        result = rag_service._invoke_with_retry(mock_invokable, {"input": "test"}, max_retries=2, initial_delay=0.01)

        assert result["error"] == "429"
        assert "surchargé" in result["answer"]

    def test_invoke_with_retry_non_429_error(self, rag_service):
        """Test that non-429 errors are raised immediately."""
        mock_invokable = Mock()
        mock_invokable.invoke.side_effect = ValueError("Some other error")

        with pytest.raises(ValueError, match="Some other error"):
            rag_service._invoke_with_retry(mock_invokable, {"input": "test"})

    def test_invoke_with_retry_exponential_backoff(self, rag_service):
        """Test exponential backoff timing."""
        mock_invokable = Mock()
        expected_result = {"answer": "Success"}

        # Raise 429 twice, then succeed
        mock_invokable.invoke.side_effect = [
            Exception("429: Too Many Requests"),
            Exception("429: Too Many Requests"),
            expected_result,
        ]

        with patch("time.sleep") as mock_sleep:
            result = rag_service._invoke_with_retry(
                mock_invokable,
                {"input": "test"},
                max_retries=4,
                initial_delay=1,
            )

            assert result == expected_result
            # Check exponential backoff: 1s, 2s, 4s
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)  # 2^0 * 1
            mock_sleep.assert_any_call(2)  # 2^1 * 1


class TestCreateRagChain:
    """Test RAG chain creation."""

    @patch("app.services.rag_service.create_retrieval_chain")
    @patch("app.services.rag_service.create_stuff_documents_chain")
    @patch("app.services.rag_service.ChatPromptTemplate")
    @patch("app.services.rag_service.get_rag_prompt")
    def test_create_rag_chain_success(
        self,
        mock_get_prompt,
        mock_prompt_template,
        mock_stuff_chain,
        mock_retrieval_chain,
        rag_service,
    ):
        """Test successful RAG chain creation."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_get_prompt.return_value = "Test prompt"
        mock_prompt = Mock()
        mock_prompt_template.from_template.return_value = mock_prompt
        mock_stuff = Mock()
        mock_stuff_chain.return_value = mock_stuff
        mock_chain = Mock()
        mock_retrieval_chain.return_value = mock_chain

        result = rag_service._create_rag_chain(mock_vector_store)

        assert result == mock_chain
        mock_vector_store.as_retriever.assert_called_once()
        # Verify retriever k parameter
        call_kwargs = mock_vector_store.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 6

    @patch("app.services.rag_service.create_retrieval_chain")
    @patch("app.services.rag_service.create_stuff_documents_chain")
    @patch("app.services.rag_service.ChatPromptTemplate")
    @patch("app.services.rag_service.get_rag_prompt")
    def test_create_rag_chain_uses_llm(
        self,
        mock_get_prompt,
        mock_prompt_template,
        mock_stuff_chain,
        mock_retrieval_chain,
        rag_service,
    ):
        """Test that RAG chain uses initialized LLM."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_get_prompt.return_value = "Test prompt"
        mock_prompt = Mock()
        mock_prompt_template.from_template.return_value = mock_prompt

        with patch.object(rag_service, "_get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            rag_service._create_rag_chain(mock_vector_store)

            mock_get_llm.assert_called_once()
            # Verify LLM is passed to stuff_chain
            assert mock_stuff_chain.call_args[0][0] == mock_llm


class TestRebuildIndex:
    """Test index rebuild functionality."""

    @patch("app.services.rag_service.IndexManager")
    @patch("app.services.rag_service.FAISS")
    @patch("app.services.rag_service.DocumentBuilder")
    @patch("app.services.rag_service.fetch_all_events")
    def test_rebuild_index_success(
        self,
        mock_fetch,
        mock_builder_class,
        mock_faiss,
        mock_index_manager_class,
        rag_service,
    ):
        """Test successful index rebuild."""
        # Mock event fetching
        mock_event = Mock()
        mock_fetch.return_value = [mock_event]

        # Mock document building
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_doc = Mock()
        mock_builder.build.return_value = [mock_doc]

        # Mock embedding provider
        mock_embeddings = Mock()
        with patch.object(rag_service, "_get_or_create_embedding_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_provider.get_embeddings.return_value = mock_embeddings
            mock_provider.get_distance_strategy.return_value = "COSINE"
            mock_get_provider.return_value = mock_provider

            # Mock vector store
            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 1
            mock_faiss.from_documents.return_value = mock_vector_store

            # Mock index manager
            mock_index_manager = Mock()
            mock_index_manager_class.return_value = mock_index_manager

            # Mock RAG chain creation
            with patch.object(rag_service, "_create_rag_chain") as mock_create_chain:
                mock_chain = Mock()
                mock_create_chain.return_value = mock_chain

                result = rag_service.rebuild_index("mistral")

                assert result["status"] == "success"
                assert result["provider"] == "mistral"
                assert rag_service.vector_stores["mistral"] == mock_vector_store
                assert rag_service.rag_chains["mistral"] == mock_chain

    @patch("app.services.rag_service.fetch_all_events")
    def test_rebuild_index_invalid_provider(self, mock_fetch, rag_service):
        """Test rebuild with invalid provider."""
        result = rag_service.rebuild_index("invalid_provider")

        assert result["status"] == "error"
        assert "Unsupported provider" in result["message"]

    @patch("app.services.rag_service.fetch_all_events")
    def test_rebuild_index_fetch_error(self, mock_fetch, rag_service):
        """Test rebuild when fetching events fails."""
        mock_fetch.side_effect = Exception("API error")

        result = rag_service.rebuild_index("mistral")

        assert result["status"] == "error"
        assert "API error" in result["message"]


class TestLoadIndex:
    """Test index loading functionality."""

    @patch("app.services.rag_service.IndexManager")
    def test_load_index_success(self, mock_index_manager_class, rag_service):
        """Test successful index load."""
        # Mock embedding provider
        mock_embeddings = Mock()
        with patch.object(rag_service, "_get_or_create_embedding_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_provider.get_embeddings.return_value = mock_embeddings
            mock_get_provider.return_value = mock_provider

            # Mock index manager
            mock_index_manager = Mock()
            mock_vector_store = Mock()
            mock_index_manager.load_index.return_value = mock_vector_store
            mock_index_manager.get_index_info.return_value = {"status": "ready"}
            mock_index_manager_class.return_value = mock_index_manager

            # Mock RAG chain creation
            with patch.object(rag_service, "_create_rag_chain") as mock_create_chain:
                mock_chain = Mock()
                mock_create_chain.return_value = mock_chain

                result = rag_service.load_index("mistral")

                assert result["status"] == "success"
                assert result["provider"] == "mistral"
                assert rag_service.vector_stores["mistral"] == mock_vector_store
                assert rag_service.rag_chains["mistral"] == mock_chain

    @patch("app.services.rag_service.IndexManager")
    def test_load_index_not_found(self, mock_index_manager_class, rag_service):
        """Test load when index doesn't exist."""
        mock_embeddings = Mock()
        with patch.object(rag_service, "_get_or_create_embedding_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_provider.get_embeddings.return_value = mock_embeddings
            mock_get_provider.return_value = mock_provider

            # Mock index manager returning None
            mock_index_manager = Mock()
            mock_index_manager.load_index.return_value = None
            mock_index_manager_class.return_value = mock_index_manager

            result = rag_service.load_index("mistral")

            assert result["status"] == "not_found"
            assert "/rebuild first" in result["message"]

    def test_load_index_invalid_provider(self, rag_service):
        """Test load with invalid provider."""
        result = rag_service.load_index("invalid_provider")

        assert result["status"] == "error"
        assert "Unsupported provider" in result["message"]


class TestAnswerQuestionExtended:
    """Extended tests for answer_question method."""

    @patch("app.services.rag_service.RAGService.classify_intent")
    @patch("app.services.rag_service.RAGService.get_rag_chain")
    @patch("app.services.rag_service.RAGService._invoke_with_retry")
    def test_answer_question_rag_with_error_result(self, mock_invoke, mock_get_chain, mock_classify, rag_service):
        """Test RAG response with error from invoke."""
        mock_classify.return_value = INTENT_RAG
        mock_chain = Mock()
        mock_get_chain.return_value = mock_chain
        rag_service.rag_chains["mistral"] = mock_chain

        mock_invoke.return_value = {
            "error": "429",
            "answer": "Rate limited",
        }

        result = rag_service.answer_question("Test query", provider="mistral")

        assert result["status"] == "success"
        assert result["answer"] == "Rate limited"
        assert result["events"] == []

    @patch("app.services.rag_service.RAGService.classify_intent")
    @patch("app.services.rag_service.RAGService.get_rag_chain")
    @patch("app.services.rag_service.RAGService._invoke_with_retry")
    def test_answer_question_rag_with_context_extraction(self, mock_invoke, mock_get_chain, mock_classify, rag_service):
        """Test proper context extraction from RAG response."""
        mock_classify.return_value = INTENT_RAG
        mock_chain = Mock()
        mock_get_chain.return_value = mock_chain
        rag_service.rag_chains["mistral"] = mock_chain

        # Create mock document with page_content
        mock_doc = Mock()
        mock_doc.page_content = "Event information content"
        mock_doc.metadata = {
            "originagenda_title": "Concert",
            "location_city": "Pau",
        }

        mock_invoke.return_value = {
            "answer": "Voici les événements",
            "context": [mock_doc],
        }

        result = rag_service.answer_question("Find events", provider="mistral")

        assert result["status"] == "success"
        assert len(result["context"]) == 1
        assert result["context"][0] == "Event information content"

    def test_answer_question_exception_handling(self, rag_service):
        """Test exception handling in answer_question."""
        with patch.object(rag_service, "classify_intent") as mock_classify:
            mock_classify.side_effect = Exception("Unexpected error")

            result = rag_service.answer_question("Test")

            assert result["status"] == "error"
            assert "Erreur lors du traitement" in result["answer"]
            assert result["events"] == []
            assert result["context"] == []
