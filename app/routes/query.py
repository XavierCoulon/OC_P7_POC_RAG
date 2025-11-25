"""Query endpoint for RAG."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"], dependencies=[Depends(verify_api_key)])


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str


class EventInfo(BaseModel):
    """Event information from RAG context."""

    title: str
    location: str
    start_date: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    question: str
    answer: str
    intent: str = "RAG"  # RAG or CHAT
    provider: str = "mistral"  # Embedding provider used
    events: List[EventInfo] = []  # Source events for RAG responses
    context: List[str] = []  # Raw text chunks from knowledge base used for answer


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request_body: QueryRequest,
    request: Request,
    embedding_provider: str = Query("mistral", enum=["mistral", "huggingface"]),
):
    """Ask a question to the RAG system with intent classification.

    This endpoint:
    1. Classifies the query intent (RAG or CHAT)
    2. If CHAT: Returns a friendly message
    3. If RAG: Uses the vector index to answer (using specified embedding provider)

    Query Parameters:
        - embedding_provider: Embedding provider to use for RAG ("mistral" or "huggingface")

    Args:
        request_body: Query request with question
        request: FastAPI request object
        embedding_provider: Which embedding provider to use

    Returns:
        QueryResponse with answer, intent, provider, and source events
    """
    try:
        rag_service = request.app.state.rag_service

        # Use answer_question which handles classification and routing
        result = rag_service.answer_question(request_body.question, provider=embedding_provider)

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("answer"))

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            intent=result["intent"],
            provider=result.get("provider", embedding_provider),
            events=[
                EventInfo(
                    title=event["title"],
                    location=event["location"],
                    start_date=event["start_date"],
                    url=event.get("url"),
                )
                for event in result.get("events", [])
            ],
            context=result.get("context", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
