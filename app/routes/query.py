"""Query endpoint for RAG."""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import logging

from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"], dependencies=[Depends(verify_api_key)])


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    question: str
    answer: str
    intent: str = "RAG"  # RAG or CHAT


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request_body: QueryRequest, request: Request):
    """Ask a question to the RAG system with intent classification.

    This endpoint:
    1. Classifies the query intent (RAG or CHAT)
    2. If CHAT: Returns a friendly message
    3. If RAG: Uses the vector index to answer

    Args:
        request_body: Query request with question
        request: FastAPI request object

    Returns:
        QueryResponse with answer and intent
    """
    try:
        rag_service = request.app.state.rag_service

        # Use answer_question which handles classification and routing
        result = rag_service.answer_question(request_body.question)

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("answer"))

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            intent=result["intent"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
