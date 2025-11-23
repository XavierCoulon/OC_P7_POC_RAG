"""Rebuild endpoint for RAG index."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rebuild"], dependencies=[Depends(verify_api_key)])


class RebuildResponse(BaseModel):
    """Response model for rebuild endpoint."""

    status: str
    provider: str = ""
    message: str
    metadata: dict = {}


class IndexInfoResponse(BaseModel):
    """Response model for index info endpoint."""

    status: str
    provider: str = ""
    message: str = ""
    metadata: dict = {}


class ProvidersStatusResponse(BaseModel):
    """Response model for providers status endpoint."""

    status: str
    providers: dict = {}


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_rag_index(request: Request, provider: str = Query("mistral", enum=["mistral", "huggingface"])):
    """Rebuild the RAG index from OpenAgenda API for a specific provider.

    This fetches all events, creates embeddings using the specified provider,
    and builds a FAISS vector store. The index is saved to disk for later use.

    Query Parameters:
        - provider: Embedding provider to use ("mistral" or "huggingface")

    Returns:
        RebuildResponse with status and metadata
    """
    try:
        rag_service = request.app.state.rag_service
        result = rag_service.rebuild_index(provider=provider)

        return RebuildResponse(
            status=result["status"],
            provider=result.get("provider", provider),
            message=result["message"],
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Rebuild endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


@router.get("/index/info", response_model=IndexInfoResponse)
async def get_index_information(request: Request, provider: str = Query("mistral", enum=["mistral", "huggingface"])):
    """Get information about the current index for a specific provider.

    Query Parameters:
        - provider: Embedding provider ("mistral" or "huggingface")

    Returns:
        IndexInfoResponse with index metadata
    """
    try:
        rag_service = request.app.state.rag_service
        metadata = rag_service.get_index_info(provider=provider)

        status = metadata.get("status", "unknown")
        if status == "not_initialized":
            message = f"No index loaded for {provider}. Call /rebuild?provider={provider} first."
        elif status == "not_found":
            message = "No index found on disk. Call /rebuild first."
        elif status == "available":
            message = "Index is ready for queries."
        else:
            message = ""

        return IndexInfoResponse(status=status, message=message, metadata=metadata)

    except Exception as e:
        logger.error(f"Index info endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting index info: {str(e)}")


@router.get("/providers/status", response_model=ProvidersStatusResponse)
async def get_providers_status(request: Request):
    """Get status of all embedding providers.

    Returns:
        ProvidersStatusResponse with status for each provider
    """
    try:
        rag_service = request.app.state.rag_service
        providers_info = rag_service.get_available_providers()

        return ProvidersStatusResponse(
            status="success",
            providers=providers_info,
        )

    except Exception as e:
        logger.error(f"Get providers status error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting providers status: {str(e)}")
