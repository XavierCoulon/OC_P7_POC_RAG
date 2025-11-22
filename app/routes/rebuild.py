"""Rebuild endpoint for RAG index."""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import logging

from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rebuild"], dependencies=[Depends(verify_api_key)])


class RebuildResponse(BaseModel):
    """Response model for rebuild endpoint."""

    status: str
    message: str
    metadata: dict = {}


class IndexInfoResponse(BaseModel):
    """Response model for index info endpoint."""

    status: str
    message: str = ""
    metadata: dict = {}


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_rag_index(request: Request):
    """Rebuild the RAG index from OpenAgenda API.

    This fetches all events, creates embeddings, and builds a FAISS vector store.
    The index is saved to disk for later use.

    Returns:
        RebuildResponse with status and metadata
    """
    try:
        rag_service = request.app.state.rag_service
        result = rag_service.rebuild_index()

        return RebuildResponse(
            status=result["status"],
            message=result["message"],
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Rebuild endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


@router.get("/index/info", response_model=IndexInfoResponse)
async def get_index_information(request: Request):
    """Get information about the current index.

    Returns:
        IndexInfoResponse with index metadata
    """
    try:
        rag_service = request.app.state.rag_service
        metadata = rag_service.get_index_info()

        status = metadata.get("status", "unknown")
        if status == "not_initialized":
            message = "No index loaded. Call /rebuild first."
        elif status == "not_found":
            message = "No index found on disk. Call /rebuild first."
        elif status == "available":
            message = "Index is ready for queries."
        else:
            message = ""

        return IndexInfoResponse(status=status, message=message, metadata=metadata)

    except Exception as e:
        logger.error(f"Index info endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting index info: {str(e)}"
        )
