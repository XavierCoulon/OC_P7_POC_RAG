"""Health check endpoint."""

import os
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
    location_department: str
    first_date: str


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
        location_department=os.getenv("LOCATION_DEPARTMENT", "Pyrénées-Atlantiques"),
        first_date=os.getenv("FIRST_DATE", "2025-01-01T00:00:00"),
    )
