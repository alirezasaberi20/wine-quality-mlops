"""Wine Quality Classification - Source Module"""

# Lazy imports to avoid importing mlflow in production
# Import directly when needed:
#   from src.preprocess import WineDataPreprocessor
#   from src.train import WineModelTrainer
#   from src.predict import WinePredictor

__all__ = ["WineDataPreprocessor", "WineModelTrainer", "WinePredictor"]
