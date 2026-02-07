"""Model training module with MLflow tracking for Wine Quality Classification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.preprocess import WineDataPreprocessor


class WineModelTrainer:
    """Handles model training, evaluation, and MLflow logging."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.model_type = self.config["model"]["type"]
        self.preprocessor = WineDataPreprocessor(config_path)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
    
    def _create_model(self) -> Any:
        """Create model based on configuration.
        
        Returns:
            Sklearn model instance.
        """
        if self.model_type == "random_forest":
            params = self.config["model"]["random_forest"]
            return RandomForestClassifier(**params)
        elif self.model_type == "logistic_regression":
            params = self.config["model"]["logistic_regression"]
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters from configuration (for logging).
        
        Returns:
            Dictionary of model parameters.
        """
        params = self.config["model"][self.model_type].copy()
        params["model_type"] = self.model_type
        return params
    
    def train(
        self, X_train: np.ndarray, y_train: pd.Series
    ) -> Any:
        """Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            Trained model.
        """
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        print(f"Model trained: {self.model_type}")
        return self.model
    
    def evaluate(
        self, X_test: np.ndarray, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1_score": f1_score(y_test, y_pred, average="binary"),
        }
        
        print("\n" + "=" * 50)
        print("Model Evaluation Results")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def train_with_mlflow(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        run_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, float], str]:
        """Train model with MLflow tracking.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.
            run_name: Optional name for the MLflow run.
            
        Returns:
            Tuple of (trained model, metrics dict, run_id).
        """
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"\nMLflow Run ID: {run_id}")
            
            # Log parameters
            params = self._get_model_params()
            params["quality_threshold"] = self.config["data"]["quality_threshold"]
            mlflow.log_params(params)
            
            # Log data info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Train model
            self.train(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name="wine_quality_classifier"
            )
            
            # Log feature importance for tree-based models
            if hasattr(self.model, "feature_importances_"):
                feature_importance = dict(
                    zip(self.config["features"], self.model.feature_importances_)
                )
                # Log as artifact
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=["feature", "importance"]
                ).sort_values("importance", ascending=False)
                
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                
                print("\nFeature Importance (Top 5):")
                print(importance_df.head())
            
            print(f"\nMLflow tracking complete. View at: mlruns/")
            
        return self.model, metrics, run_id
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save trained model to disk.
        
        Args:
            path: Path to save model. Uses config path if not provided.
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if path is None:
            path = self.config["artifacts"]["model_path"]
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: Optional[str] = None) -> Any:
        """Load model from disk.
        
        Args:
            path: Path to load model from. Uses config path if not provided.
            
        Returns:
            Loaded model.
        """
        if path is None:
            path = self.config["artifacts"]["model_path"]
        
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Wine Quality Classification - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = WineModelTrainer()
    
    # Preprocess data
    print("\n[Step 1] Preprocessing data...")
    X_train, X_test, y_train, y_test, _ = trainer.preprocessor.preprocess_pipeline()
    
    # Train with MLflow tracking
    print("\n[Step 2] Training model with MLflow tracking...")
    model, metrics, run_id = trainer.train_with_mlflow(
        X_train, X_test, y_train, y_test,
        run_name="wine_quality_training"
    )
    
    # Save model and scaler
    print("\n[Step 3] Saving artifacts...")
    trainer.save_model()
    trainer.preprocessor.save_scaler()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {trainer.config['artifacts']['model_path']}")
    print(f"Scaler saved to: {trainer.config['artifacts']['scaler_path']}")
    print(f"MLflow Run ID: {run_id}")
    print(f"\nTo view MLflow UI, run: mlflow ui")
    
    return model, metrics


if __name__ == "__main__":
    main()
