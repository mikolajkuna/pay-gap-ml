from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

class AutoGluonPredictor:
    """
    AutoGluon predictor class for tabular regression.
    Loads trained predictor and evaluates on test data,
    including counterfactual gender pay gap analysis.
    """

    def __init__(
        self,
        predictor: TabularPredictor,
        data_path: Path,
        feature_columns: list[str],
    ) -> None:
        self.predictor = predictor
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.df_test: pd.DataFrame | None = None
        self.predictions: np.ndarray | None = None

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path, sep=";")
        self.df_test = df.dropna(subset=self.feature_columns + ["income"])

    def predict(self) -> np.ndarray:
        """
        Runs model predictions on loaded test data and returns np.ndarray.
        """
        if self.df_test is None:
            raise RuntimeError("Test data not loaded.")

        self.predictions = self.predictor.predict(self.df_test)
        return self.predictions.to_numpy()

    def evaluate_metrics(self) -> dict[str, float]:
        """
        Returns regression metrics (MAE, CV, means) on test dataset.
        """
        if self.predictions is None:
            raise RuntimeError("No predictions available.")

        y_true = self.df_test["income"].values
        y_pred = self.predictions
        mae = np.mean(np.abs(y_true - y_pred))
        cv = np.std(y_true - y_pred) / np.mean(y_true) * 100
        return {
            "MAE_test": mae,
            "CV_test": cv,
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
        }

    def evaluate_counterfactual_gap(self) -> dict[str, float]:
        """
        Computes counterfactual gender pay gap and simple mean gap.
        """
        if self.df_test is None or self.predictions is None:
            raise RuntimeError("Data not loaded or predictions not run.")

        df = self.df_test.copy()
        X = df[self.feature_columns]

        # counterfactual male/female
        X_cf_m = X.copy()
        X_cf_f = X.copy()
        X_cf_m["gender"] = 1
        X_cf_f["gender"] = 0

        pred_m = self.predictor.predict(X_cf_m).to_numpy()
        pred_f = self.predictor.predict(X_cf_f).to_numpy()

        gap_pln = pred_m - pred_f
        gap_pct = gap_pln / pred_m * 100

        # simple mean gap
        mean_m = df[df["gender"] == 1]["income"].mean()
        mean_f = df[df["gender"] == 0]["income"].mean()
        simple_gap_pln = mean_m - mean_f
        simple_gap_pct = simple_gap_pln / mean_m * 100

        return {
            "counterfactual_mean_pln": float(np.mean(gap_pln)),
            "counterfactual_mean_pct": float(np.mean(gap_pct)),
            "counterfactual_median_pln": float(np.median(gap_pln)),
            "counterfactual_median_pct": float(np.median(gap_pct)),
            "simple_mean_pln": float(simple_gap_pln),
            "simple_mean_pct": float(simple_gap_pct),
        }


def main() -> None:
    test_path = Path("data/processed/salary_data_synthetic.csv")
    features = [
        "age", "gender", "education_level", "job_level",
        "experience_years", "distance_from_home", "absence", "child"
    ]

    # Load a previously trained predictor
    predictor = TabularPredictor.load("AutogluonModels/ag-20250101_0000")  # adjust path

    auto_predictor = AutoGluonPredictor(
        predictor=predictor,
        data_path=test_path,
        feature_columns=features,
    )
    auto_predictor.load_data()
    predictions = auto_predictor.predict()

    evaluation = auto_predictor.evaluate_metrics()
    gap_results = auto_predictor.evaluate_counterfactual_gap()

    print("\n--- AutoGluon test metrics ---")
    print(evaluation)

    print("\n--- AutoGluon gender pay gap ---")
    print(gap_results)


if __name__ == "__main__":
    main()
