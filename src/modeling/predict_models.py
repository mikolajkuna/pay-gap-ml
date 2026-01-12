from joblib import load
import xgboost as xgb
from utils import load_csv, preprocess, counterfactual_gap
from sklearn.metrics import mean_absolute_error
import shap
from autogluon.tabular import TabularPredictor

train_path = "data/salary_data_train.csv"
synthetic_path = "data/salary_data_synthetic.csv"

test_df = preprocess(load_csv(synthetic_path))
features = ["age", "gender", "education_level", "job_level", "experience_years", "distance_from_home", "absence", "child"]
X_test = test_df[features].astype(float)
y_test = test_df["income"].astype(float)

models = {
    "LR": load("models/lr_model.joblib"),
    "RF": load("models/rf_model.joblib"),
    "XGB": xgb.XGBRegressor(),
    "Bayes": load("models/bayes_model.joblib"),
    "BART": load("models/bart_model.joblib"),
    "TabPFN": load("models/tabpfn_model.joblib"),
    "AutoGluon": TabularPredictor.load("models/autogluon_model")
}

# Wczytanie XGB
models["XGB"].load_model("models/xgb_model.json")

for name, model in models.items():
    if name == "AutoGluon":
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    gap_pln, gap_pct = counterfactual_gap(model, X_test)
    
    print(f"\n--- {name} ---")
    print(f"MAE: {mae:,.0f} PLN, Średnie income pred: {preds.mean():,.0f}")
    print(f"Średnia luka kontrfaktyczna: {gap_pln.mean():,.0f} PLN, {gap_pct.mean():.2f}%")

    # SHAP (opcjonalnie dla mniejszych próbek)
    X_shap = X_test.sample(n=150, random_state=42)
    explainer = shap.Explainer(model.predict, X_shap)
    shap_values = explainer(X_shap)
    shap.summary_plot(shap_values, X_shap, plot_type="bar")
