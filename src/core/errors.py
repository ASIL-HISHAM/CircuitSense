"""
CircuitSense — Custom Exception Classes & Error Handling
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class CircuitSenseException(Exception):
    """Base exception for CircuitSense."""
    pass


class ModelLoadError(CircuitSenseException):
    """Raised when model loading fails."""
    pass


class ModelNotFoundError(CircuitSenseException):
    """Raised when model checkpoint is not found."""
    pass


class InvalidWaveformError(CircuitSenseException):
    """Raised when waveform data is invalid."""
    pass


class PredictionError(CircuitSenseException):
    """Raised when prediction fails."""
    pass


class ConfigurationError(CircuitSenseException):
    """Raised when configuration is invalid."""
    pass


def model_not_loaded_error() -> HTTPException:
    """Return HTTP 503 when model is not loaded."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model not loaded. Please check server startup logs."
    )


def invalid_waveform_error(message: str) -> HTTPException:
    """Return HTTP 400 for invalid waveform."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid waveform: {message}"
    )


def internal_error(message: str) -> HTTPException:
    """Return HTTP 500 for internal server error."""
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal server error: {message}"
    )


def batch_too_large_error(max_size: int) -> HTTPException:
    """Return HTTP 413 for batch size exceeded."""
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"Batch size exceeds maximum of {max_size}"
    )


class ErrorResponse:
    """Standardized error response format."""
    
    def __init__(
        self,
        error: str,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.message = message
        self.code = code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error": self.error,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }
