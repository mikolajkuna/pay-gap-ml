# src/modeling/train_bayes.py

from pathlib import Path
from sklearn.linear_model import BayesianRidge
import joblib  # <- używamy joblib do zapisu modelu
from typing import List

class BayesTrainer:
    def __init__(self, max_iter=300, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6):
        self.max_iter = max_iter
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = None

    def train(self, X, y):
        """Trening modelu BayesianRidge"""
        self.model = BayesianRidge(
            max_iter=self.max_iter,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            compute_score=True
        )
        self.model.fit(X, y)
        return self.model

    def save(self, path: Path):
        """Zapisanie wytrenowanego modelu do pliku za pomocą joblib"""
        if self.model is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            print(f"[INFO] Model saved to {path}")
