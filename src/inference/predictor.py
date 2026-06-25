"""
CircuitSense — Inference Engine
Loads a trained model and predicts fault class from a waveform.
"""
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..models import CircuitSenseModel
from ..data.simulator import FAULT_CLASSES

logger = logging.getLogger(__name__)


class CircuitSensePredictor:
    """Loads trained model checkpoint and makes predictions on waveforms."""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Initialize predictor with a trained checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            device: Device to load model on ('auto', 'cuda', 'cpu', 'mps')
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            KeyError: If checkpoint format is invalid
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        try:
            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = {"cfg", "model_state"}
            if not required_keys.issubset(ckpt.keys()):
                raise KeyError(f"Checkpoint missing keys: {required_keys - set(ckpt.keys())}")
            
            self.cfg = ckpt["cfg"]
            self.model = CircuitSenseModel(self.cfg["model"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            
            self.threshold = self.cfg.get("inference", {}).get("threshold", 0.6)
            logger.info(f"✓ Model loaded. Confidence threshold: {self.threshold}")
            
        except KeyError as e:
            raise KeyError(f"Invalid checkpoint format: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {type(e).__name__}: {str(e)}")
            raise

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """Normalize and convert waveform to model input tensor.
        
        Args:
            waveform: Raw waveform array (1024 samples)
            
        Returns:
            Tensor ready for model (B=1, C=1, L=1024)
        """
        w = waveform.astype(np.float32)
        # Normalize: zero mean, unit variance
        w_mean = w.mean()
        w_std = w.std()
        if w_std < 1e-8:
            w_std = 1e-8  # Avoid division by zero
        w = (w - w_mean) / w_std
        
        return torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> Dict:
        """Predict fault class from a waveform.
        
        Args:
            waveform: Input waveform array (1024 samples)
            
        Returns:
            Dict with predicted_class, fault_name, confidence, probabilities, 
            attention_weights, and is_confident flag
        """
        x = self.preprocess(waveform)
        probs, attn_w = self.model.predict_proba(x)
        probs_np = probs.cpu().numpy()[0]
        
        pred_class = int(probs_np.argmax())
        confidence = float(probs_np.max())
        
        return {
            "predicted_class": pred_class,
            "fault_name": FAULT_CLASSES.get(pred_class, "UNKNOWN"),
            "confidence": confidence,
            "all_probabilities": {
                FAULT_CLASSES.get(i, f"CLASS_{i}"): float(p)
                for i, p in enumerate(probs_np)
            },
            "attention_weights": attn_w.cpu().numpy()[0].mean(axis=0).tolist(),
            "is_confident": confidence >= self.threshold,
        }

    @torch.no_grad()
    def predict_batch(self, waveforms: List[np.ndarray]) -> List[Dict]:
        """Predict fault classes for multiple waveforms.
        
        Args:
            waveforms: List of waveform arrays
            
        Returns:
            List of prediction dicts
        """
        return [self.predict(w) for w in waveforms]
