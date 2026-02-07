"""FastAPI application for Wine Quality Prediction."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import yaml

from src.predict import WinePredictor, get_sample_wine_features


# Load configuration
config_path = project_root / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (lazy loading)
predictor: Optional[WinePredictor] = None


def get_predictor() -> WinePredictor:
    """Get or initialize the predictor."""
    global predictor
    if predictor is None:
        predictor = WinePredictor(str(config_path))
    return predictor


# Pydantic models for request/response
class WineFeatures(BaseModel):
    """Wine features for prediction."""
    
    fixed_acidity: float = Field(..., description="Fixed acidity (g/dm³)", ge=0, le=20)
    volatile_acidity: float = Field(..., description="Volatile acidity (g/dm³)", ge=0, le=2)
    citric_acid: float = Field(..., description="Citric acid (g/dm³)", ge=0, le=2)
    residual_sugar: float = Field(..., description="Residual sugar (g/dm³)", ge=0, le=20)
    chlorides: float = Field(..., description="Chlorides (g/dm³)", ge=0, le=1)
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide (mg/dm³)", ge=0, le=100)
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide (mg/dm³)", ge=0, le=300)
    density: float = Field(..., description="Density (g/cm³)", ge=0.9, le=1.1)
    pH: float = Field(..., description="pH value", ge=2, le=5)
    sulphates: float = Field(..., description="Sulphates (g/dm³)", ge=0, le=3)
    alcohol: float = Field(..., description="Alcohol content (%)", ge=5, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with original feature names."""
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.pH,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    prediction: int = Field(..., description="Binary prediction (0=bad, 1=good)")
    quality_label: str = Field(..., description="Quality label (good/bad)")
    probability_bad: float = Field(..., description="Probability of bad quality")
    probability_good: float = Field(..., description="Probability of good quality")
    confidence: float = Field(..., description="Prediction confidence")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    samples: List[WineFeatures] = Field(..., description="List of wine samples")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total_samples: int = Field(..., description="Total number of samples processed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    scaler_loaded: bool = Field(..., description="Whether scaler is loaded")


class FeatureInfoResponse(BaseModel):
    """Feature information response."""
    
    features: Dict[str, Dict[str, str]] = Field(..., description="Feature information")


# API Endpoints
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Wine Quality Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    pred = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=pred.model is not None,
        scaler_loaded=pred.scaler is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: WineFeatures) -> PredictionResponse:
    """
    Make a single wine quality prediction.
    
    - **Input**: Wine physicochemical properties
    - **Output**: Quality prediction with probabilities
    """
    try:
        pred = get_predictor()
        result = pred.predict(features.to_dict())
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make batch wine quality predictions.
    
    - **Input**: List of wine samples with physicochemical properties
    - **Output**: List of quality predictions with probabilities
    """
    try:
        pred = get_predictor()
        features_list = [sample.to_dict() for sample in request.samples]
        results = pred.predict_batch(features_list)
        return BatchPredictionResponse(
            predictions=results,
            total_samples=len(results)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/features", response_model=FeatureInfoResponse, tags=["Information"])
async def get_features() -> FeatureInfoResponse:
    """Get information about required wine features."""
    pred = get_predictor()
    return FeatureInfoResponse(features=pred.get_feature_info())


@app.get("/sample", response_model=WineFeatures, tags=["Information"])
async def get_sample() -> Dict[str, float]:
    """Get sample wine features for testing."""
    sample = get_sample_wine_features()
    # Convert to API format (underscores instead of spaces)
    return {
        "fixed_acidity": sample["fixed acidity"],
        "volatile_acidity": sample["volatile acidity"],
        "citric_acid": sample["citric acid"],
        "residual_sugar": sample["residual sugar"],
        "chlorides": sample["chlorides"],
        "free_sulfur_dioxide": sample["free sulfur dioxide"],
        "total_sulfur_dioxide": sample["total sulfur dioxide"],
        "density": sample["density"],
        "pH": sample["pH"],
        "sulphates": sample["sulphates"],
        "alcohol": sample["alcohol"]
    }


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )
