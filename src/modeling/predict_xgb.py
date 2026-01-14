# src/modeling/predict_xgb.py

from pathlib import Path
import xgboost as xgb
import pandas as pd
from src.features import get_feature_columns

class XGBPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

    def predict(self, df: pd.DataFrame):
        X = df[get_feature_columns()]
        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix)
        return preds
