"""Tests for training module."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import WineDataPreprocessor
from src.train import WineModelTrainer


class TestWineDataPreprocessor:
    """Tests for WineDataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return WineDataPreprocessor()
    
    def test_load_data(self, preprocessor):
        """Test data loading."""
        df = preprocessor.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "quality" in df.columns
    
    def test_create_binary_target(self, preprocessor):
        """Test binary target creation."""
        df = preprocessor.load_data()
        df = preprocessor.create_binary_target(df)
        assert "quality_label" in df.columns
        assert set(df["quality_label"].unique()).issubset({0, 1})
    
    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling."""
        df = preprocessor.load_data()
        df = preprocessor.handle_missing_values(df)
        assert df.isnull().sum().sum() == 0
    
    def test_get_features_and_target(self, preprocessor):
        """Test feature and target extraction."""
        df = preprocessor.load_data()
        df = preprocessor.create_binary_target(df)
        X, y = preprocessor.get_features_and_target(df)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X.columns) == len(preprocessor.feature_columns)
    
    def test_split_data(self, preprocessor):
        """Test data splitting."""
        df = preprocessor.load_data()
        df = preprocessor.create_binary_target(df)
        X, y = preprocessor.get_features_and_target(df)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
    
    def test_scaler_fit_transform(self, preprocessor):
        """Test scaler fitting and transformation."""
        df = preprocessor.load_data()
        df = preprocessor.create_binary_target(df)
        X, y = preprocessor.get_features_and_target(df)
        X_train, X_test, _, _ = preprocessor.split_data(X, y)
        
        X_train_scaled = preprocessor.fit_scaler(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
    
    def test_preprocess_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline."""
        X_train, X_test, y_train, y_test, df = preprocessor.preprocess_pipeline()
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(df, pd.DataFrame)


class TestWineModelTrainer:
    """Tests for WineModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        return WineModelTrainer()
    
    @pytest.fixture
    def preprocessed_data(self, trainer):
        """Get preprocessed data."""
        return trainer.preprocessor.preprocess_pipeline()
    
    def test_create_model(self, trainer):
        """Test model creation."""
        model = trainer._create_model()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
    
    def test_train(self, trainer, preprocessed_data):
        """Test model training."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data
        
        model = trainer.train(X_train, y_train)
        assert model is not None
        assert trainer.model is not None
    
    def test_evaluate(self, trainer, preprocessed_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test, _ = preprocessed_data
        
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # Check metric ranges
        for metric, value in metrics.items():
            assert 0 <= value <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
