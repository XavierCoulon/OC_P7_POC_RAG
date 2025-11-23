"""Index manager for FAISS vector store persistence."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages FAISS index persistence."""

    def __init__(self, index_dir: str):
        """Initialize index manager.

        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Files created by FAISS.save_local()
        self.faiss_index_path = self.index_dir / "index.faiss"
        self.docstore_path = self.index_dir / "index.pkl"
        self.index_mapping_path = self.index_dir / "index.pkl"  # Same as docstore
        self.metadata_path = self.index_dir / "metadata.json"

    def save_index(self, vector_store: FAISS, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save FAISS index to disk.

        Args:
            vector_store: FAISS vector store to save
            metadata: Optional metadata to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save FAISS index components
            vector_store.save_local(str(self.index_dir))

            # Save metadata
            if metadata is None:
                metadata = {}

            metadata.setdefault("total_vectors", vector_store.index.ntotal)
            metadata.setdefault("distance_strategy", "COSINE")

            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"✓ Index saved to {self.index_dir}")
            logger.info(f"  - Total vectors: {vector_store.index.ntotal}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to save index: {e}")
            return False

    def load_index(self, embeddings: MistralAIEmbeddings) -> Optional[FAISS]:
        """Load FAISS index from disk.

        Args:
            embeddings: MistralAI embeddings instance for the index

        Returns:
            FAISS vector store or None if not found
        """
        try:
            if not self._index_exists():
                logger.warning("Index not found on disk")
                return None

            # Load FAISS index
            vector_store = FAISS.load_local(str(self.index_dir), embeddings, allow_dangerous_deserialization=True)

            # Load and log metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.info(f"✓ Index loaded from {self.index_dir}")
                logger.info(f"  - Total vectors: {metadata.get('total_vectors')}")
                logger.info(f"  - Distance: {metadata.get('distance_strategy')}")

            return vector_store

        except Exception as e:
            logger.error(f"✗ Failed to load index: {e}")
            return None

    def _index_exists(self) -> bool:
        """Check if index files exist."""
        return self.faiss_index_path.exists() and self.docstore_path.exists()

    def clear_index(self) -> bool:
        """Delete all index files.

        Returns:
            True if successful, False otherwise
        """
        try:
            for file in self.index_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info(f"✓ Index cleared from {self.index_dir}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to clear index: {e}")
            return False

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the stored index.

        Returns:
            Dictionary with index information
        """
        if not self.metadata_path.exists():
            return {"status": "not_found"}

        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["status"] = "available"
            return metadata
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return {"status": "error", "message": str(e)}
