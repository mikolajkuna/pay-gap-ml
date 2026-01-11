import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from ISLP.bart import BART
import shap

train_path = r"C:\Users\MKuna\OneDrive - Unit4\Desktop\salary_data_train.csv"
synthetic_path = r"C:\Users\MKuna\OneDrive - Unit4\Desktop\salary_data_synthetic.csv"

def load_salary_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ',' if ',' in first_line else None
    if sep is None:
        raise ValueError(f"Nie moge wykryc separatora CSV w {path}")
    df = pd.read_csv(path, sep=sep)
    expected_cols = ["age", "gender", "education_level", "job_level", "experience_years", "income", "child", "distance_from_home", "absence"]
    if df.shape[1] != len(expected_cols):
        df.columns = expected_cols
    return df

def preprocess(df):
    gender_map = {"M": 1, "Male": 1, "m": 1, "F": 0, "Female": 0, "f": 0}
    df["gender"] = df["gender"].replace(gender_map).astype(float)
    edu_map = {1: 1, 2: 2, 3: 3, 4: 4, "Bachelor's": 1, "Master's": 2, "PhD": 3, "Other": 4}
    df["education_level"] = df["education_level"].replace(edu_map).astype(float)
    job_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    df["job_level"] = df["job_level"].replace(job_map).astype(float)
    for col in ["age", "experience_years", "income", "child", "distance_from_home", "absence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_train = preprocess(load_salary_csv(train_path))
df_synth = preprocess(load_salary_csv(synthetic_path))

key_cols = ["age", "gender", "education_level", "job_level", "experience_years", "income", "child", "distance_from_home", "absence"]
df_train = df_train.dropna(subset=key_cols)
df_synth = df_synth.dropna(subset=key_cols)

features = ["age", "gender", "education_level", "job_level", "experience_years", "child", "distance_from_home", "absence"]
X_train = df_train[features].astype(np.float32)
y_train = df_train["income"].astype(np.float32)
X_synth = df_synth[features].astype(np.float32)
y_synth = df_synth["income"].astype(np.float32)

bart = BART(num_trees=150, num_particles=8, max_stages=4000, burnin=300, random_state=42, n_jobs=-1)
bart.fit(X_train, y_train)
preds_test = bart.predict(X_synth)
mae = mean_absolute_error(y_synth, preds_test)
cv = np.std(y_synth - preds_test) / np.mean(y_synth) * 100

print(f"\nMAE test: {mae:,.0f} PLN")
print(f"CV test: {cv:.2f}%")
print(f"Średnie income (true): {y_synth.mean():,.0f} PLN")
print(f"Średnie income (pred): {preds_test.mean():,.0f} PLN")

X_cf = X_synth.copy()
X_cf["gender"] = 1 - X_cf["gender"]
pred_cf = bart.predict(X_cf)
gap_pln = preds_test - pred_cf
gap_pct = gap_pln / preds_test * 100

print("\n--- Luka płacowa (kontrfaktyczna, ML) ---")
print(f"Średnia luka: {np.mean(gap_pln):,.0f} PLN")
print(f"Średnia luka (%): {np.mean(gap_pct):.2f}%")
print(f"Mediana luka: {np.median(gap_pln):,.0f} PLN")
print(f"Mediana luka (%): {np.median(gap_pct):.2f}%")

mean_m = y_synth[X_synth["gender"] == 1].mean()
mean_f = y_synth[X_synth["gender"] == 0].mean()
simple_gap_pln = mean_m - mean_f
simple_gap_pct = simple_gap_pln / mean_m * 100

print("\n--- Porównanie gender pay gap ---")
print(f"Prosty (średnie): {simple_gap_pln:,.0f} PLN, {simple_gap_pct:.2f}%")
print(f"Kontrfaktyczny (ML): {np.mean(gap_pln):,.0f} PLN, {np.mean(gap_pct):.2f}%")
print(f"Kontrfaktyczny mediana (ML): {np.median(gap_pln):,.0f} PLN, {np.median(gap_pct):.2f}%")

X_shap = X_synth.sample(n=150, random_state=42)
features_of_interest = ["gender", "child", "education_level", "job_level"]
X_shap_reduced = X_shap[features_of_interest]

def predict_wrapper(X_reduced):
    X_full = X_synth.loc[X_reduced.index].copy()
    X_full[features_of_interest] = X_reduced.values
    return bart.predict(X_full)

explainer = shap.Explainer(predict_wrapper, X_shap_reduced)
shap_values = explainer(X_shap_reduced)

shap.summary_plot(shap_values, X_shap_reduced, plot_type="bar")
shap.summary_plot(shap_values, X_shap_reduced)

for feat in features_of_interest:
    idx = features_of_interest.index(feat)
    print(f"Średni wpływ {feat} (SHAP, PLN): {shap_values.values[:, idx].mean():,.0f}")
