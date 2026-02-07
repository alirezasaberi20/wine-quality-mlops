"""Prediction utilities for Wine Quality Classification."""

import numpy as np
import pandas as pd
import joblib
import yaml
from typing import Dict, List, Union, Optional, Any
from pathlib import Path


class WinePredictor:
    """Handles predictions using trained model and scaler."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize predictor with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = None
        self.feature_columns = self.config["features"]
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model and scaler from disk."""
        model_path = self.config["artifacts"]["model_path"]
        scaler_path = self.config["artifacts"]["scaler_path"]
        
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
        
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}")
    
    def _validate_features(self, features: Dict[str, float]) -> None:
        """Validate that all required features are present.
        
        Args:
            features: Dictionary of feature names and values.
            
        Raises:
            ValueError: If required features are missing.
        """
        missing = set(self.feature_columns) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for prediction.
        
        Args:
            features: Dictionary of feature names and values.
            
        Returns:
            Scaled features as numpy array.
        """
        self._validate_features(features)
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([features])[self.feature_columns]
        
        # Scale features
        if self.scaler is not None:
            return self.scaler.transform(df)
        return df.values
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make a single prediction.
        
        Args:
            features: Dictionary of feature names and values.
            
        Returns:
            Dictionary with prediction and probability.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train model first.")
        
        X = self._prepare_features(features)
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "quality_label": "good" if prediction == 1 else "bad",
            "probability_bad": float(probability[0]),
            "probability_good": float(probability[1]),
            "confidence": float(max(probability)),
        }
    
    def predict_batch(
        self, features_list: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries.
            
        Returns:
            List of prediction dictionaries.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train model first.")
        
        # Prepare all features
        all_features = []
        for features in features_list:
            self._validate_features(features)
            all_features.append([features[col] for col in self.feature_columns])
        
        df = pd.DataFrame(all_features, columns=self.feature_columns)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(df)
        else:
            X = df.values
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "quality_label": "good" if pred == 1 else "bad",
                "probability_bad": float(prob[0]),
                "probability_good": float(prob[1]),
                "confidence": float(max(prob)),
            })
        
        return results
    
    def get_feature_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about required features.
        
        Returns:
            Dictionary with feature descriptions and typical ranges.
        """
        feature_info = {
            "fixed acidity": {
                "description": "Most acids involved with wine (tartaric acid)",
                "unit": "g/dm³",
                "typical_range": "4.6 - 15.9"
            },
            "volatile acidity": {
                "description": "Amount of acetic acid in wine",
                "unit": "g/dm³",
                "typical_range": "0.12 - 1.58"
            },
            "citric acid": {
                "description": "Found in small quantities, adds freshness",
                "unit": "g/dm³",
                "typical_range": "0 - 1"
            },
            "residual sugar": {
                "description": "Sugar remaining after fermentation",
                "unit": "g/dm³",
                "typical_range": "0.9 - 15.5"
            },
            "chlorides": {
                "description": "Amount of salt in the wine",
                "unit": "g/dm³",
                "typical_range": "0.012 - 0.611"
            },
            "free sulfur dioxide": {
                "description": "Free form of SO2 that prevents microbial growth",
                "unit": "mg/dm³",
                "typical_range": "1 - 72"
            },
            "total sulfur dioxide": {
                "description": "Total amount of SO2 (free + bound forms)",
                "unit": "mg/dm³",
                "typical_range": "6 - 289"
            },
            "density": {
                "description": "Density of wine",
                "unit": "g/cm³",
                "typical_range": "0.9901 - 1.0037"
            },
            "pH": {
                "description": "Describes acidity on scale of 0-14",
                "unit": "pH",
                "typical_range": "2.74 - 4.01"
            },
            "sulphates": {
                "description": "Wine additive contributing to SO2 levels",
                "unit": "g/dm³",
                "typical_range": "0.33 - 2.0"
            },
            "alcohol": {
                "description": "Percent alcohol content",
                "unit": "%",
                "typical_range": "8.4 - 14.9"
            }
        }
        return feature_info


def get_sample_wine_features() -> Dict[str, float]:
    """Get sample wine features for testing.
    
    Returns:
        Dictionary with sample feature values.
    """
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }


if __name__ == "__main__":
    # Test prediction
    predictor = WinePredictor()
    
    # Single prediction
    sample = get_sample_wine_features()
    print("\nSample wine features:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
    
    result = predictor.predict(sample)
    print("\nPrediction result:")
    print(f"  Quality: {result['quality_label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    # Batch prediction
    batch = [sample, sample]
    batch_results = predictor.predict_batch(batch)
    print(f"\nBatch prediction ({len(batch_results)} samples):")
    for r in batch_results:
        print(f"  Sample {r['index']}: {r['quality_label']} ({r['confidence']:.2%})")
