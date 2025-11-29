"""API configuration and settings."""

import os
from pathlib import Path


class Settings:
    """Application settings."""

    app_name: str = "Events RAG API"
    app_version: str = "0.1.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API settings
    api_title: str = "Events RAG Chatbot API"
    api_description: str = "RAG-based chatbot API for French events"
    api_version: str = "0.1.0"

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Mistral API settings
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")

    # Embedding settings
    default_embedding_provider: str = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "mistral")
    huggingface_model_name: str = os.getenv("HUGGINGFACE_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

    # Data paths
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    index_dir: Path = data_dir / "index"

    def __init__(self):
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        # Ensure data directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
