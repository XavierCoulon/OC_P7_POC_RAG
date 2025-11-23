"""Tests for query endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.rag_service import RAGService


@pytest.fixture
def client():
    """Create test client with correct API key."""
    with patch("app.core.security.API_KEY", "test-api-key"):
        app = create_app()
        yield TestClient(app)


class TestQueryEndpoint:
    """Test /ask endpoint."""

    def test_ask_missing_api_key(self, client):
        """Test that missing API key returns 403."""
        response = client.post(
            "/ask",
            json={"question": "Bonjour"},
        )
        assert response.status_code == 403
        assert "API key missing" in response.json()["detail"]

    def test_ask_invalid_api_key(self, client):
        """Test that invalid API key returns 403."""
        response = client.post(
            "/ask",
            headers={"X-API-Key": "invalid-key"},
            json={"question": "Bonjour"},
        )
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_ask_missing_question(self, client):
        """Test that missing question returns 422."""
        response = client.post(
            "/ask",
            headers={"X-API-Key": "test-api-key"},
            json={},
        )
        assert response.status_code == 422

    def test_ask_invalid_provider(self, client):
        """Test that invalid provider is rejected."""
        response = client.post(
            "/ask?embedding_provider=invalid",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "Test"},
        )
        # Invalid provider returns either 422 (validation) or 500 (runtime error)
        assert response.status_code in (422, 500)

    def test_ask_chat_intent_with_mock(self, client):
        """Test CHAT intent response with mocked service."""
        with patch.object(RAGService, "answer_question") as mock_answer:
            mock_answer.return_value = {
                "status": "success",
                "question": "Bonjour",
                "answer": "Bonjour! Comment puis-je vous aider?",
                "intent": "CHAT",
                "provider": "mistral",
                "events": [],
            }

            response = client.post(
                "/ask?embedding_provider=mistral",
                headers={"X-API-Key": "test-api-key"},
                json={"question": "Bonjour"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["intent"] == "CHAT"
            assert data["provider"] == "mistral"

    def test_ask_rag_intent_with_events_mock(self, client):
        """Test RAG intent response with events."""
        with patch.object(RAGService, "answer_question") as mock_answer:
            mock_answer.return_value = {
                "status": "success",
                "question": "Quels concerts?",
                "answer": "Plusieurs concerts disponibles",
                "intent": "RAG",
                "provider": "mistral",
                "events": [
                    {
                        "title": "Concert",
                        "location": "Pau",
                        "start_date": "2025-06-01",
                        "url": "https://example.com",
                    }
                ],
            }

            response = client.post(
                "/ask?embedding_provider=mistral",
                headers={"X-API-Key": "test-api-key"},
                json={"question": "Quels concerts?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["intent"] == "RAG"
            assert len(data["events"]) == 1
            assert data["events"][0]["title"] == "Concert"

    def test_ask_response_structure(self, client):
        """Test response structure matches schema."""
        with patch.object(RAGService, "answer_question") as mock_answer:
            mock_answer.return_value = {
                "status": "success",
                "question": "Question?",
                "answer": "Réponse",
                "intent": "CHAT",
                "provider": "mistral",
                "events": [],
            }

            response = client.post(
                "/ask",
                headers={"X-API-Key": "test-api-key"},
                json={"question": "Question?"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check required fields
            assert "question" in data
            assert "answer" in data
            assert "intent" in data
            assert "provider" in data
            assert "events" in data
            assert isinstance(data["events"], list)
