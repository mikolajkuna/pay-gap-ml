from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
import shap

from src.dataset import load_and_preprocess
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, FEATURES
from autogluon.tabular import TabularPredictor

def main():
    train_path = PROCESSED_DATA_DIR / "salary_data_train.csv"
    test_path = PROCESSED_DATA_DIR / "salary_data_synthetic.csv"

    train_df, test_df = load_and_preprocess(train_path, test_path)

    X_train = train_df[FEATURES]
    y_train = train_df["income"]
    X_test = test_df[FEATURES]
    y_test = test_df["income"]

    predictor = TabularPredictor.load(MODEL_DIR / "autogluon_model")
    preds = predictor.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    cv = np.std(y_test - preds) / np.mean(y_test) * 100

    print(f"\n--- AutoGluon ---")
    print(f"MAE test: {mae:,.0f} PLN")
    print(f"CV test: {cv:.2f}%")
    print(f"Średnie income (true): {y_test.mean():,.0f} PLN")
    print(f"Średnie income (pred): {preds.mean():,.0f} PLN")

    # ------------------ Kontrfaktyczne ------------------
    X_cf = X_test.copy()
    X_cf["gender"] = 1 - X_cf["gender"]
    gap_pln = preds - predictor.predict(X_cf)
    gap_pct = gap_pln / preds * 100

    print(f"\nŚrednia luka (PLN): {np.mean(gap_pln):,.0f}")
    print(f"Średnia luka (%): {np.mean(gap_pct):.2f}")
    print(f"Mediana luka (PLN): {np.median(gap_pln):,.0f}")
    print(f"Mediana luka (%): {np.median(gap_pct):.2f}")

    mean_m = y_test[X_test["gender"] == 1].mean()
    mean_f = y_test[X_test["gender"] == 0].mean()
    simple_gap_pln = mean_m - mean_f
    simple_gap_pct = simple_gap_pln / mean_m * 100

    print(f"\nProsty gap (średnie): {simple_gap_pln:,.0f} PLN, {simple_gap_pct:.2f}%")
    print(f"Kontrfaktyczny gap (ML): {np.mean(gap_pln):,.0f} PLN, {np.mean(gap_pct):.2f}%")
    print(f"Kontrfaktyczny mediana (ML): {np.median(gap_pln):,.0f} PLN, {np.median(gap_pct):.2f}%")

    # ------------------ SHAP ------------------
    X_shap = X_test.sample(n=150, random_state=42)
    features_of_interest = ["gender", "child", "education_level", "job_level"]
    X_shap_reduced = X_shap[features_of_interest]

    def predict_wrapper(X_reduced):
        X_full = X_test.loc[X_reduced.index].copy()
        X_full[features_of_interest] = X_reduced.values
        return predictor.predict(X_full)

    explainer = shap.Explainer(predict_wrapper, X_shap_reduced)
    shap_values = explainer(X_shap_reduced)

    shap.summary_plot(shap_values, X_shap_reduced, plot_type="bar")
    shap.summary_plot(shap_values, X_shap_reduced)

if __name__ == "__main__":
    main()
