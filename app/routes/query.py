"""Query endpoint for RAG."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, field_validator
from pydantic.fields import Field

from app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"], dependencies=[Depends(verify_api_key)])


class QueryRequest(BaseModel):
    """Modèle de requête pour l'endpoint de requête."""

    question: str = Field(
        ...,
        description="Question de l'utilisateur sur les événements culturels",
        examples=["Quels événements culturels sont proposés à Pau en 2025?"],
    )

    @field_validator("question")
    def validate_question(cls, value):
        if not value or not value.strip():
            raise ValueError("La question ne peut pas être vide.")
        return value


class EventInfo(BaseModel):
    """Informations sur un événement extrait du contexte RAG."""

    title: str = Field(..., description="Titre/nom de l'événement", examples=["Exposition sur l'océan"])
    location: str = Field(..., description="Localisation de l'événement avec code postal", examples=["Anglet (64600)"])
    start_date: str = Field(
        ..., description="Date et heure de début de l'événement au format ISO", examples=["2025-09-19 16:00:00+00:00"]
    )
    url: Optional[str] = Field(
        None, description="URL de l'événement sur OpenAgenda", examples=["https://openagenda.com/events/123456"]
    )


class QueryResponse(BaseModel):
    """Modèle de réponse pour l'endpoint de requête."""

    question: str = Field(
        ...,
        description="La question originale posée par l'utilisateur",
        examples=["Quels événements culturels sont proposés à Pau?"],
    )
    answer: str = Field(
        ...,
        description="Réponse générée par le RAG ou par le système de chat",
        examples=["Pau accueille plusieurs événements culturels..."],
    )
    intent: str = Field(
        "RAG", description="Classification de l'intention de la requête (RAG ou CHAT)", examples=["RAG"]
    )
    provider: str = Field("mistral", description="Fournisseur d'embeddings utilisé pour le RAG", examples=["mistral"])
    events: List[EventInfo] = Field(
        default_factory=list,
        description="Liste des événements sources extraits du contexte RAG",
        examples=[
            [
                {
                    "title": "Festival Peaux à Pau",
                    "location": "Pau (Pyrénées-Atlantiques)",
                    "start_date": "2025-06-22 08:00:00+00:00",
                    "url": (
                        "https://openagenda.com/eteculturel-2025-nouvelle-aquitaine/"
                        "events/festival-peaux-a-pau-9391103"
                    ),
                },
                {
                    "title": "Visite guidée : le pôle culturel du Foirail",
                    "location": "Pau (Pyrénées-Atlantiques)",
                    "start_date": "2025-09-20 07:00:00+00:00",
                    "url": (
                        "https://openagenda.com/jep-2025-nouvelle-aquitaine/"
                        "events/visite-guidee-le-pole-culturel-du-foirail"
                    ),
                },
            ]
        ],
    )
    context: List[str] = Field(
        default_factory=list,
        description="Fragments de texte brut de la base de connaissances utilisés pour générer la réponse",  # noqa: E501
        examples=[
            [
                (
                    "Titre: Festival Peaux à Pau Lieu: Pau 64000, Place récaborde, "
                    "Pau (64000), Pyrénées-Atlantiques Dates (Texte): 22 - 29 juin "
                    "Dates (ISO): Début: 2025-06-22 08:00:00+00:00 / "
                    "Fin: 2025-06-29 21:30:00+00:00 Description: "
                    "Le festival Peaux à Pau est une biennale culturelle dédiée à la "
                    "musique, mettant particulièrement l'accent sur les percussions, "
                    "un instrument fédérateur par excellence. Nous proposons une "
                    "riche et à … Détails: Le festival Peaux à Pau est une biennale "
                    "culturelle dédiée à la musique, mettant particulièrement l'accent "
                    "sur les percussions, un instrument fédérateur par excellence. "
                    "Nous proposons une programmation riche et éclectique qui vise à "
                    "rassembler des artistes de divers horizons : internationaux, "
                    "locaux, étudiants, élèves, musiciens amateurs mais aussi des "
                    "jeunes, des seniors et des habitants Notre objectif est de faire "
                    "découvrir la musique sous toutes ses formes, en offrant des "
                    "ateliers, des stages et des master-class qui sensibilisent le "
                    "public à la richesse culturelle du monde. Le festival Peaux à "
                    "Pau se positionne comme un véritable lieu de rencontre et "
                    "d'échange à travers l'art et la culture"
                )
            ]
        ],
    )


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
