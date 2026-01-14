# src/modeling/predict_rf.py
from pathlib import Path
import joblib
import pandas as pd

class RFPredictor:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
