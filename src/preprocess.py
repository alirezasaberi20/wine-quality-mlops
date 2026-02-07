"""Data preprocessing module for Wine Quality Classification."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import joblib
from pathlib import Path
from typing import Tuple, Optional


class WineDataPreprocessor:
    """Handles data loading, preprocessing, and feature engineering for wine quality data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.feature_columns = self.config["features"]
        self.target_column = self.config["target"]
        self.quality_threshold = self.config["data"]["quality_threshold"]
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load wine quality dataset from CSV.
        
        Args:
            data_path: Optional path to CSV file. Uses config path if not provided.
            
        Returns:
            DataFrame with wine quality data.
        """
        if data_path is None:
            data_path = self.config["data"]["raw_data_path"]
        
        # UCI wine quality dataset uses semicolon as delimiter
        df = pd.read_csv(data_path, sep=";")
        print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
        return df
    
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert quality score to binary classification (good/bad).
        
        Args:
            df: DataFrame with quality column.
            
        Returns:
            DataFrame with binary quality_label column.
        """
        df = df.copy()
        df["quality_label"] = (df[self.target_column] >= self.quality_threshold).astype(int)
        print(f"Quality distribution: {df['quality_label'].value_counts().to_dict()}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with missing values handled.
        """
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            # Fill numerical columns with median
            for col in self.feature_columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
        else:
            print("No missing values found in dataset")
        return df
    
    def get_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from DataFrame.
        
        Args:
            df: DataFrame with features and target.
            
        Returns:
            Tuple of (features DataFrame, target Series).
        """
        X = df[self.feature_columns]
        y = df["quality_label"]
        return X, y
    
    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        test_size = self.config["data"]["test_size"]
        random_state = self.config["data"]["random_state"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and transform.
        
        Args:
            X_train: Training features DataFrame.
            
        Returns:
            Scaled training features as numpy array.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Scaled features as numpy array.
        """
        return self.scaler.transform(X)
    
    def save_scaler(self, path: Optional[str] = None) -> None:
        """Save fitted scaler to disk.
        
        Args:
            path: Path to save scaler. Uses config path if not provided.
        """
        if path is None:
            path = self.config["artifacts"]["scaler_path"]
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path: Optional[str] = None) -> None:
        """Load scaler from disk.
        
        Args:
            path: Path to load scaler from. Uses config path if not provided.
        """
        if path is None:
            path = self.config["artifacts"]["scaler_path"]
        
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
    
    def preprocess_pipeline(
        self, data_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame]:
        """Run complete preprocessing pipeline.
        
        Args:
            data_path: Optional path to CSV file.
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, original_df).
        """
        # Load and prepare data
        df = self.load_data(data_path)
        df = self.handle_missing_values(df)
        df = self.create_binary_target(df)
        
        # Split features and target
        X, y = self.get_features_and_target(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled = self.fit_scaler(X_train)
        X_test_scaled = self.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df


if __name__ == "__main__":
    # Test preprocessing pipeline
    preprocessor = WineDataPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.preprocess_pipeline()
    print(f"\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
