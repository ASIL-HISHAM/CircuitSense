from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional


class WaveformRequest(BaseModel):
    """Request model for single waveform prediction."""
    model_config = ConfigDict(populate_by_name=True)
    waveform: List[float] = Field(..., description="Waveform array with 1024 samples")
    sample_rate: Optional[int] = Field(
        default=10000,
        alias="samplerate",
        description="Sample rate in Hz"
    )


class FaultPrediction(BaseModel):
    """Response model for fault prediction."""
    predicted_class: int = Field(..., description="Predicted fault class (0-7)")
    fault_name: str = Field(..., description="Name of predicted fault")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    attention_weights: List[float] = Field(..., description="Model attention weights")
    is_confident: bool = Field(..., description="Whether prediction confidence >= threshold")


class BatchWaveformRequest(BaseModel):
    """Request model for batch waveform predictions."""
    waveforms: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of waveforms (max 1000)"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="API status: ok, degraded, or error")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="PyTorch device: cuda, cpu, mps")
    version: str = Field(default="2.0.0", description="API version")
    error: Optional[str] = Field(default=None, description="Error message if any")