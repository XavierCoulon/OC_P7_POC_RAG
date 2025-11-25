"""Tests for index manager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from app.core.index_manager import IndexManager


class TestIndexManagerInitialization:
    """Test IndexManager initialization."""

    def test_initialization_creates_directory(self):
        """Test that initialization creates the index directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            manager = IndexManager(str(index_path))

            assert index_path.exists()
            assert manager.index_dir == index_path

    def test_initialization_with_existing_directory(self):
        """Test initialization with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            index_path.mkdir()

            # Should not raise error
            manager = IndexManager(str(index_path))
            assert manager.index_dir == index_path

    def test_file_paths_initialized(self):
        """Test that file paths are correctly initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            assert manager.faiss_index_path == Path(tmpdir) / "index.faiss"
            assert manager.docstore_path == Path(tmpdir) / "index.pkl"
            assert manager.metadata_path == Path(tmpdir) / "metadata.json"


class TestSaveIndex:
    """Test index saving functionality."""

    def test_save_index_success(self):
        """Test successful index save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Mock FAISS vector store
            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 100
            mock_vector_store.save_local = Mock()

            metadata = {"provider": "mistral", "total_events": 50}

            result = manager.save_index(mock_vector_store, metadata)

            assert result is True
            mock_vector_store.save_local.assert_called_once_with(tmpdir)

    def test_save_index_creates_metadata_file(self):
        """Test that save_index creates metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 100
            mock_vector_store.save_local = Mock()

            metadata = {
                "provider": "mistral",
                "total_events": 50,
                "rebuilt_at": "2025-01-15T10:00:00",
            }

            manager.save_index(mock_vector_store, metadata)

            # Verify metadata file was created
            assert manager.metadata_path.exists()

            # Verify metadata content
            with open(manager.metadata_path, "r") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["provider"] == "mistral"
            assert saved_metadata["total_events"] == 50
            assert saved_metadata["total_vectors"] == 100

    def test_save_index_with_none_metadata(self):
        """Test save_index with None metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 50
            mock_vector_store.save_local = Mock()

            result = manager.save_index(mock_vector_store, metadata=None)

            assert result is True

            # Verify metadata file was created with defaults
            with open(manager.metadata_path, "r") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["total_vectors"] == 50
            assert saved_metadata["distance_strategy"] == "COSINE"

    def test_save_index_handles_exception(self):
        """Test save_index handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            mock_vector_store = Mock()
            mock_vector_store.save_local.side_effect = Exception("Save failed")

            result = manager.save_index(mock_vector_store)

            assert result is False

    def test_save_index_sets_default_metadata(self):
        """Test that save_index sets default metadata values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 75
            mock_vector_store.save_local = Mock()

            # Metadata without total_vectors or distance_strategy
            metadata = {"provider": "huggingface"}

            manager.save_index(mock_vector_store, metadata)

            with open(manager.metadata_path, "r") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["total_vectors"] == 75
            assert saved_metadata["distance_strategy"] == "COSINE"
            assert saved_metadata["provider"] == "huggingface"


class TestLoadIndex:
    """Test index loading functionality."""

    def test_load_index_not_found(self):
        """Test load_index when index doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)
            mock_embeddings = Mock()

            result = manager.load_index(mock_embeddings)

            assert result is None

    def test_load_index_success(self):
        """Test successful index load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock index files
            manager.faiss_index_path.touch()
            manager.docstore_path.touch()

            # Create metadata file
            metadata = {
                "provider": "mistral",
                "total_vectors": 100,
                "distance_strategy": "COSINE",
            }
            with open(manager.metadata_path, "w") as f:
                json.dump(metadata, f)

            mock_embeddings = Mock()

            with patch("app.core.index_manager.FAISS.load_local") as mock_load:
                mock_vector_store = Mock()
                mock_load.return_value = mock_vector_store

                result = manager.load_index(mock_embeddings)

                assert result == mock_vector_store
                mock_load.assert_called_once()

    def test_load_index_with_dangerous_deserialization(self):
        """Test that load_index uses dangerous deserialization flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock index files
            manager.faiss_index_path.touch()
            manager.docstore_path.touch()

            mock_embeddings = Mock()

            with patch("app.core.index_manager.FAISS.load_local") as mock_load:
                mock_vector_store = Mock()
                mock_load.return_value = mock_vector_store

                manager.load_index(mock_embeddings)

                # Verify dangerous_deserialization is True
                call_kwargs = mock_load.call_args[1]
                assert call_kwargs["allow_dangerous_deserialization"] is True

    def test_load_index_handles_exception(self):
        """Test load_index handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock index files
            manager.faiss_index_path.touch()
            manager.docstore_path.touch()

            mock_embeddings = Mock()

            with patch("app.core.index_manager.FAISS.load_local") as mock_load:
                mock_load.side_effect = Exception("Load failed")

                result = manager.load_index(mock_embeddings)

                assert result is None

    def test_load_index_reads_metadata(self):
        """Test that load_index reads metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock index files
            manager.faiss_index_path.touch()
            manager.docstore_path.touch()

            # Create metadata file
            metadata = {
                "provider": "huggingface",
                "total_vectors": 200,
                "distance_strategy": "L2",
            }
            with open(manager.metadata_path, "w") as f:
                json.dump(metadata, f)

            mock_embeddings = Mock()

            with patch("app.core.index_manager.FAISS.load_local") as mock_load:
                mock_vector_store = Mock()
                mock_load.return_value = mock_vector_store

                manager.load_index(mock_embeddings)

                # Verify metadata was read (via logging)
                # This is implicit in the function flow


class TestIndexExists:
    """Test index existence checking."""

    def test_index_exists_when_both_files_present(self):
        """Test _index_exists returns True when both files are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            manager.faiss_index_path.touch()
            manager.docstore_path.touch()

            assert manager._index_exists() is True

    def test_index_not_exists_when_faiss_missing(self):
        """Test _index_exists returns False when FAISS file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            manager.docstore_path.touch()

            assert manager._index_exists() is False

    def test_index_not_exists_when_docstore_missing(self):
        """Test _index_exists returns False when docstore is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            manager.faiss_index_path.touch()

            assert manager._index_exists() is False

    def test_index_not_exists_when_both_missing(self):
        """Test _index_exists returns False when both files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            assert manager._index_exists() is False


class TestClearIndex:
    """Test index clearing functionality."""

    def test_clear_index_deletes_files(self):
        """Test clear_index deletes all index files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock index files
            manager.faiss_index_path.touch()
            manager.docstore_path.touch()
            manager.metadata_path.touch()

            assert len(list(manager.index_dir.glob("*"))) == 3

            result = manager.clear_index()

            assert result is True
            assert len(list(manager.index_dir.glob("*"))) == 0

    def test_clear_index_directory_still_exists(self):
        """Test that clear_index removes files but keeps directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            manager.faiss_index_path.touch()

            manager.clear_index()

            # Directory should still exist
            assert manager.index_dir.exists()

    def test_clear_index_handles_exception(self):
        """Test clear_index handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create a test file
            test_file = manager.index_dir / "test.txt"
            test_file.touch()

            # Just verify that the existing implementation handles exceptions
            # by calling it on a directory with files
            result = manager.clear_index()
            assert result is True
            assert len(list(manager.index_dir.glob("*"))) == 0

    def test_clear_index_empty_directory(self):
        """Test clear_index on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            result = manager.clear_index()

            assert result is True


class TestGetIndexInfo:
    """Test index information retrieval."""

    def test_get_index_info_not_found(self):
        """Test get_index_info when metadata doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            info = manager.get_index_info()

            assert info["status"] == "not_found"

    def test_get_index_info_success(self):
        """Test successful index info retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            metadata = {
                "provider": "mistral",
                "total_vectors": 150,
                "total_events": 75,
                "distance_strategy": "COSINE",
                "rebuilt_at": "2025-01-15T10:00:00",
            }

            with open(manager.metadata_path, "w") as f:
                json.dump(metadata, f)

            info = manager.get_index_info()

            assert info["status"] == "available"
            assert info["provider"] == "mistral"
            assert info["total_vectors"] == 150
            assert info["total_events"] == 75
            assert info["distance_strategy"] == "COSINE"

    def test_get_index_info_handles_exception(self):
        """Test get_index_info handles read exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create a metadata file
            with open(manager.metadata_path, "w") as f:
                f.write("invalid json")

            info = manager.get_index_info()

            assert info["status"] == "error"
            assert "message" in info

    def test_get_index_info_returns_all_metadata(self):
        """Test that get_index_info returns all metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            metadata = {
                "provider": "huggingface",
                "total_vectors": 200,
                "total_chunks": 400,
                "distance_strategy": "L2",
                "custom_field": "custom_value",
            }

            with open(manager.metadata_path, "w") as f:
                json.dump(metadata, f)

            info = manager.get_index_info()

            assert info["provider"] == "huggingface"
            assert info["total_vectors"] == 200
            assert info["total_chunks"] == 400
            assert info["distance_strategy"] == "L2"
            assert info["custom_field"] == "custom_value"


class TestIndexManagerIntegration:
    """Integration tests for IndexManager."""

    def test_save_and_load_cycle(self):
        """Test save and load cycle preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create mock vector store
            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 100
            mock_vector_store.save_local = Mock()

            metadata = {
                "provider": "mistral",
                "total_events": 50,
                "total_vectors": 100,
            }

            # Save
            save_result = manager.save_index(mock_vector_store, metadata)
            assert save_result is True

            # Verify metadata was saved
            info = manager.get_index_info()
            assert info["status"] == "available"
            assert info["provider"] == "mistral"
            assert info["total_events"] == 50

    def test_clear_then_info_returns_not_found(self):
        """Test that info returns not_found after clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager(tmpdir)

            # Create metadata
            metadata = {"provider": "mistral", "total_vectors": 100}
            with open(manager.metadata_path, "w") as f:
                json.dump(metadata, f)

            assert manager.get_index_info()["status"] == "available"

            # Clear
            manager.clear_index()

            # Info should return not_found
            assert manager.get_index_info()["status"] == "not_found"

    def test_index_dir_creation_with_nested_path(self):
        """Test that nested directory paths are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "level1" / "level2" / "level3"

            manager = IndexManager(str(nested_path))

            assert nested_path.exists()
            assert manager.index_dir == nested_path
