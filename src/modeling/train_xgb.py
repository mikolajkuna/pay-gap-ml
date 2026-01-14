# src/modeling/train_xgb.py

from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from src.dataset import load_csv, preprocess
from src.features import get_feature_columns
from typing import List

class XGBTrainer:
    def __init__(self, params: dict = None, num_round: int = 100):
        self.params = params or {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 42
        }
        self.num_round = num_round
        self.model = None

    def train(self, X, y, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X, label=y)
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "eval")]
        self.model = xgb.train(
            self.params,
            dtrain,
            self.num_round,
            evals=evals,
            early_stopping_rounds=10 if evals else None,
            verbose_eval=10
        )
        return self.model

    def save(self, path: Path):
        if self.model is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(path))
            print(f"[INFO] Model saved to {path}")
