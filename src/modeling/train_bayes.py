# src/modeling/train_bayes.py
from pathlib import Path
from sklearn.linear_model import BayesianRidge
import joblib

class BayesTrainer:
    def __init__(self, n_iter: int = 300, alpha_1: float = 1e-6, alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6):
        self.n_iter = n_iter
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = None

    def train(self, X, y):
        self.model = BayesianRidge(
            n_iter=self.n_iter,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            compute_score=True
        )
        self.model.fit(X, y)
        return self.model

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[INFO] Model saved to {path}")
