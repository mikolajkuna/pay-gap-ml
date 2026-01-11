from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import shap

from src.dataset import load_and_preprocess
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, FEATURES

def main():
    train_path = PROCESSED_DATA_DIR / "salary_data_train.csv"
    test_path = PROCESSED_DATA_DIR / "salary_data_synthetic.csv"

    train_df, test_df = load_and_preprocess(train_path, test_path)

    X_train = train_df[FEATURES].astype(np.float32)
    y_train = train_df["income"].astype(np.float32)
    X_test = test_df[FEATURES].astype(np.float32)
    y_test = test_df["income"].astype(np.float32)

    model = joblib.load(MODEL_DIR / "lr_model.joblib")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    cv = np.std(y_test - preds) / np.mean(y_test) * 100

    print(f"MAE: {mae:,.0f} PLN, CV: {cv:.2f}%")

    # kontrfaktyczna luka płacowa
    X_cf_m = X_test.copy()
    X_cf_f = X_test.copy()
    X_cf_m["gender"] = 1
    X_cf_f["gender"] = 0

    gap_pln = model.predict(X_cf_m) - model.predict(X_cf_f)
    gap_pct = gap_pln / model.predict(X_cf_m) * 100

    print(f"Średnia luka płacowa (PLN): {np.mean(gap_pln):,.0f}")
    print(f"Średnia luka płacowa (%): {np.mean(gap_pct):.2f}")

    # SHAP dla wybranych cech
    X_shap = X_test.sample(n=150, random_state=42)
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_shap)
    features_of_interest = ["gender", "child", "education_level", "job_level"]
    X_shap_reduced = X_shap[features_of_interest]

    shap.summary_plot(
        shap_values[:, [X_train.columns.get_loc(f) for f in features_of_interest]],
        X_shap_reduced,
        plot_type="bar"
    )

if __name__ == "__main__":
    main()
