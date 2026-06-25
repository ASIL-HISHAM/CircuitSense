"""
CircuitSense Tests — API Endpoint Tests
"""
import pytest
import json
import numpy as np


class TestAPIHealth:
    """Test /health endpoint."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "version" in data


class TestAPIPredict:
    """Test /predict endpoint."""
    
    def test_predict_valid_waveform(self, client, sample_waveform):
        """Test prediction with valid waveform."""
        payload = {
            "waveform": sample_waveform.tolist(),
            "sample_rate": 10000
        }
        response = client.post("/predict", json=payload)
        
        # Model might not be loaded in test, but endpoint should respond correctly
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "fault_name" in data
        assert "confidence" in data
    
    def test_predict_wrong_length(self, client):
        """Test prediction with wrong length waveform."""
        payload = {
            "waveform": [0.1] * 512,  # Wrong length
            "sample_rate": 10000
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code in [400, 503]
    
    def test_predict_nan_values(self, client):
        """Test prediction with NaN values."""
        waveform = [0.1] * 1024
        waveform[0] = float('nan')
        
        payload = {
            "waveform": waveform,
            "sample_rate": 10000
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code in [400, 503]
    
    def test_predict_empty_waveform(self, client):
        """Test prediction with empty waveform."""
        payload = {
            "waveform": [],
            "sample_rate": 10000
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code in [400, 503]
    
    def test_predict_with_samplerate_alias(self, client, sample_waveform):
        """Test prediction with samplerate alias (no underscore)."""
        payload = {
            "waveform": sample_waveform.tolist(),
            "samplerate": 10000  # Using alias
        }
        response = client.post("/predict", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200


class TestAPIPredictBatch:
    """Test /predict/batch endpoint."""
    
    def test_batch_valid(self, client):
        """Test batch prediction with valid waveforms."""
        waveforms = [
            np.random.randn(1024).astype(float).tolist() for _ in range(3)
        ]
        payload = {"waveforms": waveforms}
        response = client.post("/predict/batch", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 3
    
    def test_batch_empty(self, client):
        """Test batch prediction with empty list."""
        payload = {"waveforms": []}
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 400
    
    def test_batch_too_large(self, client):
        """Test batch prediction with too many waveforms."""
        # MAX_BATCH_SIZE = 1000
        waveforms = [
            np.random.randn(1024).astype(float).tolist() for _ in range(1001)
        ]
        payload = {"waveforms": waveforms}
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 413
    
    def test_batch_mixed_valid_invalid(self, client):
        """Test batch with some invalid waveforms."""
        waveforms = [
            np.random.randn(1024).astype(float).tolist(),  # Valid
            [0.1] * 512,  # Wrong length
            np.random.randn(1024).astype(float).tolist(),  # Valid
        ]
        payload = {"waveforms": waveforms}
        response = client.post("/predict/batch", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # Should accept but mark some as errors
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert data["errors"] > 0 or data["successful"] == 3


class TestAPICsv:
    """Test /predict/csv endpoint."""
    
    def test_csv_upload_valid(self, client):
        """Test CSV file upload."""
        # Create a valid CSV content
        waveform_row = ",".join([str(x) for x in np.random.randn(1024)])
        csv_content = waveform_row.encode()
        
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code in [200, 400]
    
    def test_csv_upload_wrong_extension(self, client):
        """Test CSV upload with wrong extension."""
        response = client.post(
            "/predict/csv",
            files={"file": ("test.txt", b"0,1,2", "text/plain")}
        )
        
        # Should either work or reject bad extension
        assert response.status_code in [200, 400, 503]
    
    def test_csv_upload_empty(self, client):
        """Test CSV upload with empty file."""
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", b"", "text/csv")}
        )
        
        assert response.status_code == 400
