import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gender_map = {"M": 1, "F": 0, "m": 1, "Male": 1, "Female": 0}
    df["gender"] = df["gender"].map(gender_map)
    numeric_cols = [
        "age", "experience_years", "absence", "child",
        "education_level", "job_level", "distance_from_home", "income"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["income"] >= 1276) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()

def counterfactual_gap(model, X: pd.DataFrame):
    X_m = X.copy()
    X_f = X.copy()
    X_m["gender"] = 1
    X_f["gender"] = 0
    
    if isinstance(model, TabularPredictor):
        pred_m = model.predict(X_m)  
        pred_f = model.predict(X_f)  
    else:
        pred_m = model.predict(X_m) 
        pred_f = model.predict(X_f)
    
    gap_pln = pred_m - pred_f
    gap_pct = gap_pln / pred_m * 100
    return gap_pln, gap_pct
