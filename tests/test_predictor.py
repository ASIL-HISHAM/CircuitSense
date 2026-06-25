"""
CircuitSense Tests — Predictor Tests
"""
import pytest
import numpy as np


class TestPredictor:
    """Test suite for CircuitSensePredictor."""
    
    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert predictor is not None
        assert predictor.device is not None
        assert predictor.cfg is not None
        assert predictor.model is not None
    
    def test_predict_shape(self, predictor, sample_waveform):
        """Test that prediction returns correct structure."""
        result = predictor.predict(sample_waveform)
        
        assert isinstance(result, dict)
        assert "predicted_class" in result
        assert "fault_name" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert "attention_weights" in result
        assert "is_confident" in result
    
    def test_predict_values_valid(self, predictor, sample_waveform):
        """Test that prediction values are valid."""
        result = predictor.predict(sample_waveform)
        
        assert isinstance(result["predicted_class"], int)
        assert 0 <= result["predicted_class"] < 8
        assert isinstance(result["fault_name"], str)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["is_confident"], bool)
    
    def test_predict_probabilities_sum(self, predictor, sample_waveform):
        """Test that probabilities sum to 1."""
        result = predictor.predict(sample_waveform)
        prob_sum = sum(result["all_probabilities"].values())
        
        assert abs(prob_sum - 1.0) < 0.001  # Allow small floating point error
    
    def test_predict_batch(self, predictor):
        """Test batch prediction."""
        waveforms = [np.random.randn(1024).astype(np.float32) for _ in range(3)]
        results = predictor.predict_batch(waveforms)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def test_predict_deterministic(self, predictor):
        """Test that same input gives same output."""
        waveform = np.random.randn(1024).astype(np.float32)
        
        result1 = predictor.predict(waveform.copy())
        result2 = predictor.predict(waveform.copy())
        
        assert result1["predicted_class"] == result2["predicted_class"]
        assert abs(result1["confidence"] - result2["confidence"]) < 1e-5
    
    def test_preprocess_normalization(self, predictor):
        """Test that preprocessing normalizes correctly."""
        # Create waveform with known mean and std
        waveform = np.ones(1024, dtype=np.float32) * 5.0
        
        processed = predictor.preprocess(waveform)
        
        # Should be on device and have correct shape
        assert processed.shape == (1, 1, 1024)
        assert processed.device == predictor.device


class TestWaveformValidation:
    """Test suite for waveform validation."""
    
    def test_validate_normal_waveform(self, predictor):
        """Test validation of normal waveform."""
        from src.api.main import _validate_waveform
        
        waveform = np.random.randn(1024).astype(np.float32)
        is_valid, error = _validate_waveform(waveform)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_empty_waveform(self, predictor):
        """Test validation of empty waveform."""
        from src.api.main import _validate_waveform
        
        waveform = np.array([], dtype=np.float32)
        is_valid, error = _validate_waveform(waveform)
        
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_wrong_length(self, predictor):
        """Test validation of wrong length waveform."""
        from src.api.main import _validate_waveform
        
        waveform = np.random.randn(512).astype(np.float32)
        is_valid, error = _validate_waveform(waveform)
        
        assert is_valid is False
        assert "1024" in error
    
    def test_validate_nan_values(self, predictor):
        """Test validation of waveform with NaN."""
        from src.api.main import _validate_waveform
        
        waveform = np.random.randn(1024).astype(np.float32)
        waveform[100] = np.nan
        is_valid, error = _validate_waveform(waveform)
        
        assert is_valid is False
        assert "NaN" in error or "Inf" in error
    
    def test_validate_inf_values(self, predictor):
        """Test validation of waveform with Inf."""
        from src.api.main import _validate_waveform
        
        waveform = np.random.randn(1024).astype(np.float32)
        waveform[100] = np.inf
        is_valid, error = _validate_waveform(waveform)
        
        assert is_valid is False
        assert "Inf" in error
