# src/modeling/predict_tabpfn.py

import os
from pathlib import Path
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error
from src.dataset.loader import CSVLoader
import pandas as pd
from typing import Optional

class TabPFNPredictor:
    def __init__(self, feature_columns: list[str], target_column: str = "income"):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model: Optional[TabPFNRegressor] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
            token = input("Please enter your HuggingFace token (HF_TOKEN): ").strip()
            os.environ["HF_TOKEN"] = token

    def load_model(self, model_path: Path) -> None:
        self.model = TabPFNRegressor.load(model_path)

    def load_data(self, test_file: Path) -> None:
        loader = CSVLoader(test_file, self.feature_columns, self.target_column)
        self.X_test, self.y_test = loader.load()

    def predict(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def evaluate(self) -> None:
        preds = self.predict()
        mae = mean_absolute_error(self.y_test, preds)
        cv = np.std(self.y_test - preds) / np.mean(self.y_test) * 100
        print(f"MAE: {mae:,.0f} PLN | CV: {cv:.2f}% | Mean true: {self.y_test.mean():,.0f} | Mean pred: {preds.mean():,.0f}")

    def evaluate_gender_gap(self) -> None:
        X_m = self.X_test.copy()
        X_f = self.X_test.copy()
        X_m["gender"] = 1
        X_f["gender"] = 0

        pred_m = self.model.predict(X_m)
        pred_f = self.model.predict(X_f)

        gap_pln = pred_m - pred_f
        gap_pct = gap_pln / pred_m * 100

        mean_m = self.y_test[self.X_test["gender"] == 1].mean()
        mean_f = self.y_test[self.X_test["gender"] == 0].mean()
        simple_gap_pln = mean_m - mean_f
        simple_gap_pct = simple_gap_pln / mean_m * 100

        print(f"Simple gap: {simple_gap_pln:,.0f} PLN | {simple_gap_pct:.2f}%")
        print(f"Counterfactual mean: {np.mean(gap_pln):,.0f} PLN | {np.mean(gap_pct):.2f}%")
        print(f"Counterfactual median: {np.median(gap_pln):,.0f} PLN | {np.median(gap_pct):.2f}%")
