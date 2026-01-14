from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from src.dataset import load_csv, preprocess
from src.features import get_feature_columns
from typing import Optional

class LRTrainer:
    def __init__(self):
        self.model = LinearRegression()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        self.model.fit(X, y)

        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            print(f"[INFO] MAE on validation set: {mae:,.2f} PLN")

        return self.model

    def save(self, path: Path):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[INFO] Model saved to {path}")
