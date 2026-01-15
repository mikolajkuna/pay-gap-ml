from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error
from typing import Optional, List

class AutoGluonTrainer:
    def __init__(self, feature_columns: List[str], target_column: str = "income", save_path: Path = Path("models/autogluon")):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.save_path = save_path
        self.model: Optional[TabularPredictor] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.train_data: Optional[pd.DataFrame] = None

    def train(self, time_limit: int = 300, presets: str = "best_quality") -> None:
        """
        Trains AutoGluon model.
        time_limit: seconds
        presets: "best_quality", "fast_training", etc.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be set before training.")

        self.train_data = self.X_train.copy()
        self.train_data[self.target_column] = self.y_train

        self.model = TabularPredictor(label=self.target_column, path=self.save_path).fit(
            self.train_data,
            presets=presets,
            time_limit=time_limit
        )

    def evaluate_training(self) -> None:
        preds = self.model.predict(self.X_train)
        mae = mean_absolute_error(self.y_train, preds)
        cv = np.std(self.y_train - preds) / np.mean(self.y_train) * 100
        print(f"MAE: {mae:,.0f} PLN | CV: {cv:.2f}% | Mean true: {self.y_train.mean():,.0f} | Mean pred: {preds.mean():,.0f}")

    def save_model(self) -> None:
        print(f"Model saved at: {self.save_path}")
