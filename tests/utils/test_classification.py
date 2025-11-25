"""Tests for query intent classification."""

from unittest.mock import Mock

import pytest
from langchain_mistralai import ChatMistralAI

from app.core.classification import DEFAULT_INTENT, INTENT_CHAT, INTENT_RAG, classify_query_intent


class TestClassifyQueryIntent:
    """Tests for classify_query_intent function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock ChatMistralAI instance."""
        return Mock(spec=ChatMistralAI)

    def test_classify_query_intent_rag(self, mock_llm):
        """Test classification of RAG intent."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "RAG"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Where are there events in Pau?", mock_llm)

        # Assertions
        assert result == INTENT_RAG
        mock_llm.invoke.assert_called_once()

    def test_classify_query_intent_chat(self, mock_llm):
        """Test classification of CHAT intent."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "CHAT"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("How are you today?", mock_llm)

        # Assertions
        assert result == INTENT_CHAT
        mock_llm.invoke.assert_called_once()

    def test_classify_query_intent_rag_lowercase(self, mock_llm):
        """Test classification handles lowercase RAG response."""
        # Setup mock response with lowercase
        mock_response = Mock()
        mock_response.content = "rag"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Where are events?", mock_llm)

        # Assertions
        assert result == INTENT_RAG

    def test_classify_query_intent_chat_lowercase(self, mock_llm):
        """Test classification handles lowercase CHAT response."""
        # Setup mock response with lowercase
        mock_response = Mock()
        mock_response.content = "chat"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Tell me a joke", mock_llm)

        # Assertions
        assert result == INTENT_CHAT

    def test_classify_query_intent_with_whitespace(self, mock_llm):
        """Test classification handles whitespace in response."""
        # Setup mock response with extra whitespace
        mock_response = Mock()
        mock_response.content = "  RAG  \n"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Find events", mock_llm)

        # Assertions
        assert result == INTENT_RAG

    def test_classify_query_intent_unclear_response(self, mock_llm):
        """Test classification returns default for unclear response."""
        # Setup mock response with unclear content
        mock_response = Mock()
        mock_response.content = "UNKNOWN"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Some query", mock_llm)

        # Assertions
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_empty_response(self, mock_llm):
        """Test classification returns default for empty response."""
        # Setup mock response with empty content
        mock_response = Mock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_exception_handling(self, mock_llm):
        """Test classification returns default when LLM raises exception."""
        # Setup mock to raise exception
        mock_llm.invoke.side_effect = Exception("LLM error")

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_connection_error(self, mock_llm):
        """Test classification returns default on connection error."""
        # Setup mock to raise connection error
        mock_llm.invoke.side_effect = ConnectionError("API unreachable")

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_timeout_error(self, mock_llm):
        """Test classification returns default on timeout."""
        # Setup mock to raise timeout error
        mock_llm.invoke.side_effect = TimeoutError("Request timeout")

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_long_query(self, mock_llm):
        """Test classification with long query text."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "RAG"
        mock_llm.invoke.return_value = mock_response

        # Create long query
        long_query = "Where are events? " * 50

        # Test
        result = classify_query_intent(long_query, mock_llm)

        # Assertions
        assert result == INTENT_RAG
        mock_llm.invoke.assert_called_once()

    def test_classify_query_intent_special_characters(self, mock_llm):
        """Test classification with special characters in query."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "RAG"
        mock_llm.invoke.return_value = mock_response

        # Test with special characters
        query = "Où sont les événements? @#$% 🎉"
        result = classify_query_intent(query, mock_llm)

        # Assertions
        assert result == INTENT_RAG

    def test_classify_query_intent_multiple_calls(self, mock_llm):
        """Test classification with multiple sequential calls."""
        # Setup mock responses
        responses = [
            Mock(content="RAG"),
            Mock(content="CHAT"),
            Mock(content="RAG"),
        ]
        mock_llm.invoke.side_effect = responses

        # Test multiple calls
        result1 = classify_query_intent("Query 1", mock_llm)
        result2 = classify_query_intent("Query 2", mock_llm)
        result3 = classify_query_intent("Query 3", mock_llm)

        # Assertions
        assert result1 == INTENT_RAG
        assert result2 == INTENT_CHAT
        assert result3 == INTENT_RAG
        assert mock_llm.invoke.call_count == 3

    def test_classify_query_intent_rag_with_extra_text(self, mock_llm):
        """Test classification when response contains RAG with extra text."""
        # Setup mock response with extra text
        mock_response = Mock()
        mock_response.content = "RAG - event related query"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Should still return RAG (after strip and upper)
        # Note: This test documents current behavior - the function strips whitespace
        # but doesn't extract just the intent word from mixed content
        assert result == DEFAULT_INTENT  # Because "RAG - EVENT RELATED QUERY" != "RAG"

    def test_classify_query_intent_chat_with_extra_text(self, mock_llm):
        """Test classification when response contains CHAT with extra text."""
        # Setup mock response with extra text
        mock_response = Mock()
        mock_response.content = "CHAT - general conversation"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Should return default (after strip and upper)
        assert result == DEFAULT_INTENT

    def test_classify_query_intent_rag_exact_match(self, mock_llm):
        """Test classification with exact "RAG" match."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "RAG"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == INTENT_RAG

    def test_classify_query_intent_chat_exact_match(self, mock_llm):
        """Test classification with exact "CHAT" match."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "CHAT"
        mock_llm.invoke.return_value = mock_response

        # Test
        result = classify_query_intent("Query", mock_llm)

        # Assertions
        assert result == INTENT_CHAT


class TestClassificationConstants:
    """Tests for classification constants."""

    def test_intent_rag_value(self):
        """Test INTENT_RAG constant value."""
        assert INTENT_RAG == "RAG"

    def test_intent_chat_value(self):
        """Test INTENT_CHAT constant value."""
        assert INTENT_CHAT == "CHAT"

    def test_default_intent_value(self):
        """Test DEFAULT_INTENT is set to RAG."""
        assert DEFAULT_INTENT == INTENT_RAG

    def test_default_intent_prioritizes_rag(self):
        """Test that default intent prioritizes RAG (event search)."""
        # This test documents the design decision to prioritize event search
        assert DEFAULT_INTENT == "RAG"
