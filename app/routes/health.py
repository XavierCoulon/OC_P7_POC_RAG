"""Health check endpoint."""

import os

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Modèle de réponse pour le contrôle de santé."""

    status: str = Field(..., description="Statut de santé de l'API", examples=["healthy"])
    message: str = Field(..., description="Message de statut", examples=["API en cours d'exécution"])
    location_department: str = Field(
        ...,
        description="Département de localisation OpenAgenda configuré pour cette instance",
        examples=["Pyrénées-Atlantiques"],
    )
    first_date: str = Field(
        ..., description="Date de première récupération des événements (format ISO)", examples=["2025-01-01T00:00:00"]
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: Status, message, and OpenAgenda configuration.
    """
    return HealthResponse(
        status="healthy",
        message="API is running",
        location_department=os.getenv("LOCATION_DEPARTMENT", ""),
        first_date=os.getenv("FIRST_DATE", ""),
    )
