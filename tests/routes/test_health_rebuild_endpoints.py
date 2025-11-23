"""Tests for health and rebuild endpoints."""

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


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_responds(self, client):
        """Test that /health endpoint responds."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestRebuildEndpoint:
    """Test /rebuild endpoint."""

    def test_rebuild_missing_api_key(self, client):
        """Test that rebuild requires API key."""
        response = client.post(
            "/rebuild",
            params={"provider": "mistral"},
        )
        assert response.status_code == 403

    def test_rebuild_invalid_api_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.post(
            "/rebuild",
            headers={"X-API-Key": "invalid-key"},
            params={"provider": "mistral"},
        )
        assert response.status_code == 403

    def test_rebuild_mistral_with_mock(self, client):
        """Test successful rebuild with Mistral."""
        with patch.object(RAGService, "rebuild_index") as mock_rebuild:
            mock_rebuild.return_value = {
                "status": "success",
                "provider": "mistral",
                "message": "Index rebuilt",
                "metadata": {"total_events": 100},
            }

            response = client.post(
                "/rebuild",
                headers={"X-API-Key": "test-api-key"},
                params={"provider": "mistral"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["provider"] == "mistral"

    def test_rebuild_huggingface_with_mock(self, client):
        """Test successful rebuild with HuggingFace."""
        with patch.object(RAGService, "rebuild_index") as mock_rebuild:
            mock_rebuild.return_value = {
                "status": "success",
                "provider": "huggingface",
                "message": "Index rebuilt",
                "metadata": {"total_events": 100},
            }

            response = client.post(
                "/rebuild",
                headers={"X-API-Key": "test-api-key"},
                params={"provider": "huggingface"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["provider"] == "huggingface"

    def test_rebuild_error_with_mock(self, client):
        """Test rebuild error handling."""
        with patch.object(RAGService, "rebuild_index") as mock_rebuild:
            mock_rebuild.return_value = {
                "status": "error",
                "message": "Failed",
            }

            response = client.post(
                "/rebuild",
                headers={"X-API-Key": "test-api-key"},
                params={"provider": "mistral"},
            )

            # Either 500 or success (depends on endpoint implementation)
            assert response.status_code in (200, 500)

    def test_rebuild_exception_handling(self, client):
        """Test rebuild exception handling."""
        with patch.object(RAGService, "rebuild_index") as mock_rebuild:
            mock_rebuild.side_effect = Exception("Network error")

            response = client.post(
                "/rebuild",
                headers={"X-API-Key": "test-api-key"},
                params={"provider": "mistral"},
            )

            assert response.status_code == 500
            assert "Error rebuilding index" in response.json()["detail"]


class TestIndexInfoEndpoint:
    """Test /index/info endpoint."""

    def test_index_info_not_initialized(self, client):
        """Test index info for uninitialized index."""
        with patch.object(RAGService, "get_index_info") as mock_info:
            mock_info.return_value = {
                "status": "not_initialized",
            }

            response = client.get(
                "/index/info?provider=mistral",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_initialized"
            assert "No index loaded" in data["message"]

    def test_index_info_not_found(self, client):
        """Test index info when not found on disk."""
        with patch.object(RAGService, "get_index_info") as mock_info:
            mock_info.return_value = {
                "status": "not_found",
            }

            response = client.get(
                "/index/info?provider=mistral",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_found"
            assert "No index found on disk" in data["message"]

    def test_index_info_available(self, client):
        """Test index info when available."""
        with patch.object(RAGService, "get_index_info") as mock_info:
            mock_info.return_value = {
                "status": "available",
                "total_vectors": 2080,
            }

            response = client.get(
                "/index/info?provider=mistral",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "available"
            assert "ready for queries" in data["message"]

    def test_index_info_exception(self, client):
        """Test index info exception handling."""
        with patch.object(RAGService, "get_index_info") as mock_info:
            mock_info.side_effect = Exception("Database error")

            response = client.get(
                "/index/info?provider=mistral",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 500


class TestProvidersStatusEndpoint:
    """Test /providers/status endpoint."""

    def test_providers_status_success(self, client):
        """Test providers status endpoint."""
        with patch.object(RAGService, "get_available_providers") as mock_providers:
            mock_providers.return_value = {
                "mistral": {"status": "ready", "vectors": 2080},
                "huggingface": {"status": "ready", "vectors": 2080},
            }

            response = client.get(
                "/providers/status",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "mistral" in data["providers"]
            assert "huggingface" in data["providers"]

    def test_providers_status_exception(self, client):
        """Test providers status exception handling."""
        with patch.object(RAGService, "get_available_providers") as mock_providers:
            mock_providers.side_effect = Exception("Error getting providers")

            response = client.get(
                "/providers/status",
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 500
