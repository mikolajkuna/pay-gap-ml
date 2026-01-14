from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from src.utils import load_csv, preprocess
from typing import List


class BayesPredictor:
    """
    Bayesian Ridge predictor class for salary prediction.
    Handles loading test data, preprocessing, making predictions,
    and calculating counterfactual gender pay gaps.
    """

    def __init__(self, model: BayesianRidge, data_path: Path, feature_columns: List[str]) -> None:
        self.model = model
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.X_test: pd.DataFrame | None = None
        self.df_test: pd.DataFrame | None = None
        self.preds: np.ndarray | None = None

    def load_and_prepare_data(self) -> None:
        df = preprocess(load_csv(self.data_path))
        self.df_test = df
        self.X_test = df[self.feature_columns].astype(np.float32)

    def predict(self) -> np.ndarray:
        if self.X_test is None:
            raise ValueError("Test data not loaded. Call load_and_prepare_data() first.")
        self.preds = self.model.predict(self.X_test)
        return self.preds

    def evaluate_gap(self) -> dict:
        if self.preds is None or self.df_test is None:
            raise ValueError("Predictions not available. Call predict() first.")

        df = self.df_test.copy()

        # Counterfactual predictions
        X_cf_m = self.X_test.copy()
        X_cf_f = self.X_test.copy()
        X_cf_m["gender"] = 1
        X_cf_f["gender"] = 0
        pred_m = self.model.predict(X_cf_m)
        pred_f = self.model.predict(X_cf_f)

        gap_pln = pred_m - pred_f
        gap_pct = gap_pln / pred_m * 100

        # Simple mean-based gap
        mean_m = df[df["gender"] == 1]["income"].mean()
        mean_f = df[df["gender"] == 0]["income"].mean()
        simple_gap_pln = mean_m - mean_f
        simple_gap_pct = simple_gap_pln / mean_m * 100

        return {
            "counterfactual_mean_pln": np.mean(gap_pln),
            "counterfactual_mean_pct": np.mean(gap_pct),
            "counterfactual_median_pln": np.median(gap_pln),
            "counterfactual_median_pct": np.median(gap_pct),
            "simple_mean_pln": simple_gap_pln,
            "simple_mean_pct": simple_gap_pct
        }
