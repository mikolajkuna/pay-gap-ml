from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from src.dataset import load_and_preprocess
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, FEATURES


def main():
    train_path = PROCESSED_DATA_DIR / "salary_data_train.csv"
    test_path = PROCESSED_DATA_DIR / "salary_data_synthetic.csv"

    train_df, _ = load_and_preprocess(train_path, test_path)

    X_train = train_df[FEATURES]
    y_train = train_df["income"]

    train_data = train_df[FEATURES + ["income"]]


    predictor = TabularPredictor(label="income", path=str(MODEL_DIR))

    predictor.fit(train_data)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save()


if __name__ == "__main__":
    main()
