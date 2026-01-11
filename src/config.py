# src/mikolajkuna/config.py

"""
Konfiguracja projektu Pay Gap ML:
- ścieżki do danych
- foldery na zapisane modele
- inne stałe globalne
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent 

DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATA_DIR / "salary_data_train.csv"
SYNTHETIC_CSV = DATA_DIR / "salary_data_synthetic.csv"

MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "linear_regression": MODELS_DIR / "linear_regression.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "bayesian_ridge": MODELS_DIR / "bayesian_ridge.pkl",
    "bart": MODELS_DIR / "bart.pkl",
    "tabpfn": MODELS_DIR / "tabpfn.pkl"
}

RANDOM_STATE = 42
MIN_INCOME = 1276  
