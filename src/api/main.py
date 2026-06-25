"""
CircuitSense — FastAPI Server with Professional Error Handling
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import io
import csv
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

from .schemas import WaveformRequest, FaultPrediction, BatchWaveformRequest, HealthResponse
from ..core import get_logger, model_not_loaded_error, invalid_waveform_error, batch_too_large_error, internal_error
from ..inference import CircuitSensePredictor

# Configure module logger
logger = get_logger(__name__)

# Global state
predictor: Optional[CircuitSensePredictor] = None
start_time: float = time.time()
model_load_error: Optional[str] = None


async def load_model_on_startup() -> None:
    """Load the trained model checkpoint on startup."""
    global predictor, model_load_error
    
    try:
        # Determine checkpoint path
        base_dir = Path(__file__).resolve().parent.parent.parent
        checkpoint_path = base_dir / "data" / "checkpoints" / "best.pt"
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        predictor = CircuitSensePredictor(str(checkpoint_path))
        logger.info("✓ Model loaded successfully")
        
    except Exception as e:
        error_msg = f"Failed to load model: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        model_load_error = error_msg
        predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    await load_model_on_startup()
    yield
    # Shutdown (cleanup if needed)
    logger.info("Server shutting down")


app = FastAPI(
    title="CircuitSense API",
    description="Neural fault diagnosis from circuit waveforms",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration - allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Static files configuration
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_path.exists():
    logger.info(f"Serving static files from: {frontend_path}")
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
else:
    logger.warning(f"Frontend directory not found: {frontend_path}")


@app.get("/")
def root() -> FileResponse:
    """Serve the frontend index page."""
    index_file = frontend_path / "index_minimal.html"
    if index_file.exists():
        logger.info("Serving root index")
        return FileResponse(str(index_file), media_type="text/html")
    else:
        logger.error(f"Index file not found: {index_file}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index file not found"
        )


@app.get("/index_minimal.html")
def serve_index() -> FileResponse:
    """Serve the frontend index page directly."""
    index_file = frontend_path / "index_minimal.html"
    if index_file.exists():
        logger.info("Serving index_minimal.html")
        return FileResponse(str(index_file), media_type="text/html")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index file not found"
        )


def _validate_waveform(waveform: np.ndarray, expected_length: int = 1024) -> tuple[bool, Optional[str]]:
    """Validate waveform data integrity.
    
    Args:
        waveform: Numpy array of waveform values
        expected_length: Expected length of waveform
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if waveform.size == 0:
        return False, "Waveform is empty"
    
    if len(waveform) != expected_length:
        return False, f"Expected {expected_length} samples, got {len(waveform)}"
    
    if not np.isfinite(waveform).all():
        return False, "Waveform contains NaN or Inf values"
    
    try:
        waveform.astype(np.float32)
    except (ValueError, TypeError):
        return False, "Waveform values must be numeric"
    
    return True, None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API health and model status."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        device=device,
        version="2.0.0",
        error=model_load_error
    )


@app.post("/predict", response_model=FaultPrediction, status_code=status.HTTP_200_OK)
def predict(req: WaveformRequest) -> FaultPrediction:
    """Predict fault class from a single waveform.
    
    Args:
        req: WaveformRequest containing waveform array and sample rate
        
    Returns:
        FaultPrediction with predicted class and confidence
        
    Raises:
        HTTPException: If model not loaded or waveform invalid
    """
    if predictor is None:
        logger.error("Prediction attempted without loaded model")
        raise model_not_loaded_error()
    
    try:
        waveform = np.array(req.waveform, dtype=np.float32)
        is_valid, error_msg = _validate_waveform(waveform)
        
        if not is_valid:
            logger.warning(f"Invalid waveform: {error_msg}")
            raise invalid_waveform_error(error_msg)
        
        result = predictor.predict(waveform)
        return FaultPrediction(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {type(e).__name__}")
        raise internal_error(type(e).__name__)


@app.post("/predict/batch")
def predict_batch(req: BatchWaveformRequest) -> dict:
    """Predict fault classes from multiple waveforms.
    
    Args:
        req: BatchWaveformRequest with list of waveforms
        
    Returns:
        Dict with predictions list and summary
        
    Raises:
        HTTPException: If model not loaded or batch invalid
    """
    if predictor is None:
        raise model_not_loaded_error()
    
    # Batch size limit to prevent DoS
    MAX_BATCH_SIZE = 1000
    if len(req.waveforms) > MAX_BATCH_SIZE:
        raise batch_too_large_error(MAX_BATCH_SIZE)
    
    if len(req.waveforms) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Waveforms list is empty"
        )
    
    results = []
    errors = []
    
    for idx, w in enumerate(req.waveforms):
        try:
            waveform = np.array(w, dtype=np.float32)
            is_valid, error_msg = _validate_waveform(waveform)
            
            if not is_valid:
                errors.append({"index": idx, "error": error_msg})
                results.append({"error": error_msg, "index": idx})
            else:
                results.append(predictor.predict(waveform))
                
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
            results.append({"error": str(e), "index": idx})
    
    if errors:
        logger.warning(f"Batch prediction: {len(errors)}/{len(req.waveforms)} failed")
    
    return {
        "predictions": results,
        "count": len(results),
        "errors": len(errors),
        "successful": len(req.waveforms) - len(errors)
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict:
    """Upload a CSV file with waveforms (one waveform per row, 1024 columns).
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        Dict with predictions, count, and error summary
        
    Raises:
        HTTPException: If model not loaded or file invalid
    """
    if predictor is None:
        raise model_not_loaded_error()
    
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and TXT files are supported"
        )
    
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        if not text_content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        reader = csv.reader(io.StringIO(text_content))
        results = []
        row_errors = []
        
        for row_idx, row in enumerate(reader):
            if not row or all(not v.strip() for v in row):
                continue  # Skip empty rows
                
            try:
                # Parse numeric values
                values = [float(v.strip()) for v in row[:1024]]
                
                if len(values) < 1024:
                    raise ValueError(f"Row has only {len(values)} values, expected 1024")
                
                waveform = np.array(values, dtype=np.float32)
                is_valid, error_msg = _validate_waveform(waveform)
                
                if not is_valid:
                    row_errors.append({"row": row_idx, "error": error_msg})
                    results.append({"error": error_msg, "row": row_idx})
                else:
                    results.append(predictor.predict(waveform))
                    
            except ValueError as e:
                row_errors.append({"row": row_idx, "error": str(e)})
                results.append({"error": f"Invalid numeric values: {str(e)}", "row": row_idx})
            except Exception as e:
                row_errors.append({"row": row_idx, "error": str(e)})
                results.append({"error": str(e), "row": row_idx})
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid rows found in file"
            )
        
        if row_errors:
            logger.warning(f"CSV upload: {len(row_errors)}/{len(results)} rows had errors")
        
        return {
            "predictions": results,
            "count": len(results),
            "errors": len(row_errors),
            "successful": len(results) - len(row_errors)
        }
        
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding error. Please use UTF-8"
        )
    except Exception as e:
        logger.exception(f"CSV processing error: {type(e).__name__}")
        raise internal_error(type(e).__name__)
