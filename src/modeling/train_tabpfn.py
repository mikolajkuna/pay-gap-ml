from pathlib import Path
import joblib
import numpy as np
from tabpfn import TabPFNRegressor

from src.dataset import load_and_preprocess
from src.config import PROCESSED_DATA_DIR, MODEL_DIR, FEATURES

def main():
    train_path = PROCESSED_DATA_DIR / "salary_data_train.csv"
    test_path = PROCESSED_DATA_DIR / "salary_data_synthetic.csv"

    train_df, _ = load_and_preprocess(train_path, test_path)

    X_train = train_df[FEATURES].astype(np.float32)
    y_train = train_df["income"].astype(np.float32)

    model = TabPFNRegressor(random_state=42)
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "tabpfn_model.joblib")

if __name__ == "__main__":
    main()
