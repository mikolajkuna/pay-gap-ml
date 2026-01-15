# src/modeling/train_tabpfn.py

import os
from pathlib import Path
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error
from src.dataset.loader import CSVLoader
import pandas as pd
from typing import Optional

class TabPFNTrainer:
    def __init__(self, feature_columns: list[str], target_column: str = "income"):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model: Optional[TabPFNRegressor] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None

        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
            token = input("Please enter your HuggingFace token (HF_TOKEN): ").strip()
            os.environ["HF_TOKEN"] = token

    def load_data(self, train_file: Path) -> None:
        loader = CSVLoader(train_file, self.feature_columns, self.target_column)
        self.X_train, self.y_train = loader.load()

    def train(self, num_trees: int = 150, num_particles: int = 8) -> None:
        self.model = TabPFNRegressor(
            num_trees=num_trees,
            num_particles=num_particles,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_training(self) -> None:
        preds = self.model.predict(self.X_train)
        mae = mean_absolute_error(self.y_train, preds)
        cv = np.std(self.y_train - preds) / np.mean(self.y_train) * 100
        print(f"MAE: {mae:,.0f} PLN | CV: {cv:.2f}% | Mean true: {self.y_train.mean():,.0f} | Mean pred: {preds.mean():,.0f}")

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
