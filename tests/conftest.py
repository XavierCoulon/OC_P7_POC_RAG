"""Pytest configuration."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set default values for missing env vars in tests
os.environ.setdefault("MISTRAL_API_KEY", "test-key-do-not-use")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("LOCATION_DEPARTMENT", "Pyrénées-Atlantiques")
os.environ.setdefault("FIRST_DATE", "2025-01-01T00:00:00")
os.environ.setdefault("DEFAULT_EMBEDDING_PROVIDER", "mistral")
os.environ.setdefault("HUGGINGFACE_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
