from pathlib import Path
import numpy as np
import joblib

class LRPredictor:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
