# src/modeling/train_autogluon.py

from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error
from src.dataset.loader import CSVLoader
from typing import Optional

class AutoGluonTrainer:
    def __init__(self, feature_columns: list[str], target_column: str = "income", save_path: Path = Path("models/autogluon")):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.save_path = save_path
        self.model: Optional[TabularPredictor] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None

    def load_data(self, train_file: Path) -> None:
        loader = CSVLoader(train_file, self.feature_columns, self.target_column)
        self.X_train, self.y_train = loader.load()
        # AutoGluon expects target column inside the DataFrame
        self.train_data = self.X_train.copy()
        self.train_data[self.target_column] = self.y_train

    def train(self, time_limit: int = 300, presets: str = "best_quality") -> None:
        """
        time_limit: seconds for AutoGluon training
        presets: "best_quality", "fast_training", "medium_quality", etc.
        """
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
