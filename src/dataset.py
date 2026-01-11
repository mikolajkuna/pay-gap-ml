# src/mikolajkuna/dataset.py

"""
Dataset module for PayGap-ML project.

Loads raw CSV files, merges synthetic data if requested,
preprocesses data, and saves processed dataset ready for modeling.
"""

from pathlib import Path
import pandas as pd
from src.mikolajkuna import config, features


def load_and_preprocess(
    train_file: Path = config.TRAIN_CSV,
    synthetic_file: Path = config.SYNTHETIC_CSV,
    output_file: Path = config.DATA_DIR / "paygap_processed.csv",
    include_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load training and optional synthetic CSVs, preprocess, merge, and save.

    Returns:
        pd.DataFrame: Processed dataset ready for feature engineering.
    """

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    df_train = pd.read_csv(train_file, sep="," if "," in open(train_file).readline() else ";")
    print(f"[INFO] Loaded training data: {df_train.shape[0]} rows")

    # Load synthetic if requested
    if include_synthetic and synthetic_file.exists():
        df_synth = pd.read_csv(synthetic_file, sep="," if "," in open(synthetic_file).readline() else ";")
        df = pd.concat([df_train, df_synth], ignore_index=True)
        print(f"[INFO] Combined dataset: {df.shape[0]} rows")
    else:
        df = df_train

    gender_map = {"M": 1, "Male": 1, "m": 1, "F": 0, "Female": 0, "f": 0}
    df["gender"] = df["gender"].map(gender_map)


    numeric_cols = features.FEATURES + ["income"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    df = df[(df["income"] >= config.MIN_INCOME) & (df[numeric_cols].ge(0).all(axis=1))]


    df = df.dropna().reset_index(drop=True)
    print(f"[INFO] After preprocessing: {df.shape[0]} rows")


    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[INFO] Processed dataset saved to {output_file}")

    return df


if __name__ == "__main__":
    load_and_preprocess()
