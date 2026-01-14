from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from src.utils import load_csv, preprocess
from typing import List


class BayesTrainer:
    """
    Bayesian Ridge trainer class for salary prediction.
    Handles loading data, preprocessing, training, and evaluation.
    """

    def __init__(self, data_path: Path, feature_columns: List[str], target_column: str = "income") -> None:
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model: BayesianRidge | None = None
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None

    def load_and_prepare_data(self) -> None:
        df = preprocess(load_csv(self.data_path))
        self.X_train = df[self.feature_columns].astype(np.float32)
        self.y_train = df[self.target_column].astype(np.float32)

    def train_model(self) -> BayesianRidge:
        self.model = BayesianRidge()
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self) -> dict:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        preds = self.model.predict(self.X_train)
        mae = mean_absolute_error(self.y_train, preds)
        cv = np.std(self.y_train - preds) / np.mean(self.y_train) * 100
        return {
            "MAE": mae,
            "CV": cv,
            "mean_true": self.y_train.mean(),
            "mean_pred": preds.mean()
        }
