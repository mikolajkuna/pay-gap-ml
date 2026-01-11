import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt

train_path = r"C:\Users\MKuna\OneDrive - Unit4\Desktop\salary_data_train.csv"
synthetic_path = r"C:\Users\MKuna\OneDrive - Unit4\Desktop\salary_data_synthetic.csv"

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)

train_df = load_csv(train_path)
synthetic_df = load_csv(synthetic_path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    numeric_cols = [
        "age", "experience_years", "absence", "child",
        "education_level", "job_level", "distance_from_home", "income"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["income"] >= 1276) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()

train_df = preprocess(train_df)
synthetic_df = preprocess(synthetic_df)

features = [
    "age", "gender", "education_level", "job_level",
    "experience_years", "distance_from_home", "absence", "child"
]

X_train = train_df[features].astype(np.float32)
y_train = train_df["income"].astype(np.float32)

X_test = synthetic_df[features].astype(np.float32)
y_test = synthetic_df["income"].astype(np.float32)

model_xgb = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    objective="reg:squarederror"
)
model_xgb.fit(X_train, y_train)

preds = model_xgb.predict(X_test)
mae = mean_absolute_error(y_test, preds)
cv = np.std(y_test - preds) / np.mean(y_test) * 100

print(f"\n--- XGBoost ---")
print(f"MAE test: {mae:,.0f} PLN")
print(f"CV test: {cv:.2f}%")
print(f"Średnie income (true): {y_test.mean():,.0f} PLN")
print(f"Średnie income (pred): {preds.mean():,.0f} PLN")

X_cf_m = X_test.copy()
X_cf_f = X_test.copy()
X_cf_m["gender"] = 1
X_cf_f["gender"] = 0

pred_m = model_xgb.predict(X_cf_m)
pred_f = model_xgb.predict(X_cf_f)

gap_pln = pred_m - pred_f
gap_pct = gap_pln / pred_m * 100

print("\n--- Luka płacowa (kontrfaktyczna, ML) ---")
print(f"Średnia luka: {np.mean(gap_pln):,.0f} PLN")
print(f"Średnia luka (%): {np.mean(gap_pct):.2f}%")
print(f"Mediana luka: {np.median(gap_pln):,.0f} PLN")
print(f"Mediana luka (%): {np.median(gap_pct):.2f}%")

mean_m = y_test[X_test["gender"] == 1].mean()
mean_f = y_test[X_test["gender"] == 0].mean()
simple_gap_pln = mean_m - mean_f
simple_gap_pct = simple_gap_pln / mean_m * 100

print("\n--- Porównanie gender pay gap ---")
print(f"Prosty (średnie): {simple_gap_pln:,.0f} PLN, {simple_gap_pct:.2f}%")
print(f"Kontrfaktyczny (ML): {np.mean(gap_pln):,.0f} PLN, {np.mean(gap_pct):.2f}%")
print(f"Kontrfaktyczny mediana (ML): {np.median(gap_pln):,.0f} PLN, {np.median(gap_pct):.2f}%")

# ============================ SHAP ============================
X_shap = X_test.sample(n=150, random_state=42)
explainer_xgb = shap.Explainer(model_xgb)
shap_values_xgb = explainer_xgb(X_shap)

features_of_interest = ["gender", "child", "education_level", "job_level"]
X_shap_reduced = X_shap[features_of_interest]

shap.summary_plot(
    shap_values_xgb[:, [X_train.columns.get_loc(f) for f in features_of_interest]],
    X_shap_reduced,
    plot_type="bar"
)
shap.summary_plot(
    shap_values_xgb[:, [X_train.columns.get_loc(f) for f in features_of_interest]],
    X_shap_reduced
)

for f in features_of_interest:
    idx = X_train.columns.get_loc(f)
    print(f"Średni wpływ {f} (SHAP, PLN): {shap_values_xgb.values[:, idx].mean():,.0f}")
