import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error

class TabPFNPredictor:
    """
    Predictor for TabPFN regression model.
    Automatically asks for HuggingFace token if not set in environment.
    """
    def __init__(self, model: TabPFNRegressor, data_path: str, feature_columns: list):
        self.model = model
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None

        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
            token = input("Please enter your HuggingFace token (HF_TOKEN): ").strip()
            os.environ["HF_TOKEN"] = token

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path, sep=";")
        df = df.dropna(subset=self.feature_columns + ["income"])
        self.X_test = df[self.feature_columns].astype(np.float32)
        self.y_test = df["income"].astype(np.float32)

    def predict(self) -> np.ndarray:
        return self.model.predict(self.X_test)

    def evaluate_gap(self) -> None:
        X_cf_m = self.X_test.copy()
        X_cf_f = self.X_test.copy()
        X_cf_m["gender"] = 1
        X_cf_f["gender"] = 0

        pred_m = self.model.predict(X_cf_m)
        pred_f = self.model.predict(X_cf_f)

        gap_pln = pred_m - pred_f
        gap_pct = gap_pln / pred_m * 100

        mean_m = self.y_test[self.X_test["gender"] == 1].mean()
        mean_f = self.y_test[self.X_test["gender"] == 0].mean()
        simple_gap_pln = mean_m - mean_f
        simple_gap_pct = simple_gap_pln / mean_m * 100

        print("\n--- Gender pay gap evaluation ---")
        print(f"Simple (mean) gap: {simple_gap_pln:,.0f} PLN, {simple_gap_pct:.2f}%")
        print(f"Counterfactual gap (ML, mean): {np.mean(gap_pln):,.0f} PLN, {np.mean(gap_pct):.2f}%")
        print(f"Counterfactual gap (ML, median): {np.median(gap_pln):,.0f} PLN, {np.median(gap_pct):.2f}%")

def main() -> None:
    data_path = "data/processed/salary_data_synthetic.csv"
    features = [
        "age", "gender", "education_level", "job_level",
        "experience_years", "distance_from_home", "absence", "child"
    ]

    # Load pretrained model
    model = TabPFNRegressor(num_trees=150, num_particles=8, random_state=42, n_jobs=-1)
    # For simplicity, fit on train data first
    train_data = pd.read_csv("data/processed/salary_data_train.csv", sep=";")
    X_train = train_data[features].astype(np.float32)
    y_train = train_data["income"].astype(np.float32)
    model.fit(X_train, y_train)

    predictor = TabPFNPredictor(model=model, data_path=data_path, feature_columns=features)
    predictor.load_data()
    predictions = predictor.predict()
    mae = mean_absolute_error(predictor.y_test, predictions)
    cv = np.std(predictor.y_test - predictions) / np.mean(predictor.y_test) * 100

    print("\n--- TabPFN evaluation ---")
    print(f"MAE test: {mae:,.0f} PLN")
    print(f"CV test: {cv:.2f}%")
    print(f"Mean income (true): {predictor.y_test.mean():,.0f} PLN")
    print(f"Mean income (pred): {predictions.mean
