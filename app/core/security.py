"""Security and authentication utilities."""

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

# Get API key from environment
API_KEY = os.getenv("API_KEY", "default-key-change-me")

# Define the API key header requirement
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Verify the API key from the request header.

    Args:
        api_key: The API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: 403 Forbidden if API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key missing. Please provide X-API-Key header.",
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
