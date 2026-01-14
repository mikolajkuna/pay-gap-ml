# src/modeling/predict_bayes.py

import joblib
import numpy as np
import pandas as pd

class BayesPredictor:
    def __init__(self, model_path: Path):
        """
        Inicjalizacja klasy do przewidywania przy użyciu modelu BayesianRidge.
        
        :param model_path: Ścieżka do zapisany modelu
        """
        self.model = joblib.load(model_path)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dokonanie predykcji"""
        return self.model.predict(X)
