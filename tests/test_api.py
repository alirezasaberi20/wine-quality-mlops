"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample wine features for testing."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "scaler_loaded" in data
    
    def test_predict(self, client, sample_features):
        """Test single prediction endpoint."""
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "quality_label" in data
        assert "probability_bad" in data
        assert "probability_good" in data
        assert "confidence" in data
        
        assert data["prediction"] in [0, 1]
        assert data["quality_label"] in ["good", "bad"]
        assert 0 <= data["probability_bad"] <= 1
        assert 0 <= data["probability_good"] <= 1
    
    def test_predict_batch(self, client, sample_features):
        """Test batch prediction endpoint."""
        request_data = {"samples": [sample_features, sample_features]}
        response = client.post("/predict_batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total_samples" in data
        assert data["total_samples"] == 2
        assert len(data["predictions"]) == 2
    
    def test_features(self, client):
        """Test features info endpoint."""
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        
        assert "features" in data
        assert len(data["features"]) == 11  # 11 wine features
    
    def test_sample(self, client):
        """Test sample features endpoint."""
        response = client.get("/sample")
        assert response.status_code == 200
        data = response.json()
        
        # Check all required features are present
        required = [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
        ]
        for feature in required:
            assert feature in data
    
    def test_predict_invalid_features(self, client):
        """Test prediction with invalid features."""
        invalid_data = {"fixed_acidity": 7.4}  # Missing required features
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_out_of_range(self, client, sample_features):
        """Test prediction with out-of-range values."""
        sample_features["alcohol"] = 100  # Out of range
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Tests for API documentation."""
    
    def test_docs_available(self, client):
        """Test Swagger docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
