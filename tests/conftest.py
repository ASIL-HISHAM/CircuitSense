"""
CircuitSense Tests — Pytest Configuration
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_waveform():
    """Fixture: Generate a sample waveform."""
    import numpy as np
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def predictor():
    """Fixture: Initialize predictor with trained model."""
    from src.inference import CircuitSensePredictor
    try:
        return CircuitSensePredictor("data/checkpoints/best.pt")
    except FileNotFoundError:
        pytest.skip("Model checkpoint not found")


@pytest.fixture
def client():
    """Fixture: FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api import app
    return TestClient(app)
