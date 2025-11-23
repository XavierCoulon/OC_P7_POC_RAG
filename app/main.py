"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.routes import health, rebuild, query
from app.services import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown.

    Args:
        app: FastAPI application instance
    """
    # Startup - Load indices for all providers
    logger.info("Loading RAG indices on startup...")
    providers = ["mistral", "huggingface"]

    for provider in providers:
        logger.info(f"Loading index for {provider}...")
        result = rag_service.load_index(provider=provider)
        if result["status"] == "success":
            logger.info(f"✓ RAG index loaded successfully for {provider}")
        elif result["status"] == "not_found":
            logger.warning(
                f"⚠ No index found for {provider}. Call /rebuild?provider={provider} to create one."
            )
        else:
            logger.error(
                f"✗ Failed to load index for {provider}: {result.get('message')}"
            )

    yield

    # Shutdown
    logger.info("Application shutting down...")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store service in app state for dependency injection
    app.state.rag_service = rag_service

    # Include routes
    app.include_router(health.router)
    app.include_router(rebuild.router)
    app.include_router(query.router)

    logger.info(f"Application {settings.app_name} v{settings.api_version} initialized")

    return app


app = create_app()
