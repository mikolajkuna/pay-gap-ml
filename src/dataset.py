# src/mikolajkuna/dataset.py

"""
Dataset module for PayGap-ML project.

Loads raw CSV files, optionally merges synthetic data,
preprocesses data, and saves a processed dataset ready for modeling.
"""

from pathlib import Path
import pandas as pd
from src.mikolajkuna import config, features


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load CSV file and automatically detect separator (',' or ';').

    Args:
        file_path (Path): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        sep = ";" if ";" in first_line else ","

    return pd.read_csv(file_path, sep=sep)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess salary dataset:
    - map gender to numeric
    - convert numeric columns
    - filter out invalid rows

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    gender_map = {"M": 1, "Male": 1, "m": 1, "F": 0, "Female": 0, "f": 0}
    df["gender"] = df["gender"].map(gender_map)

    numeric_cols = features.FEATURES + ["income"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["income"] >= config.MIN_INCOME) & (df[numeric_cols].ge(0).all(axis=1))]
    df = df.dropna().reset_index(drop=True)
    return df


def load_and_preprocess(
    train_file: Path = config.TRAIN_CSV,
    synthetic_file: Path = config.SYNTHETIC_CSV,
    output_file: Path = config.DATA_DIR / "paygap_processed.csv",
    include_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load training and optional synthetic CSVs, preprocess, merge, and save.

    Args:
        train_file (Path): Path to training CSV.
        synthetic_file (Path): Path to synthetic CSV.
        output_file (Path): Path to save processed CSV.
        include_synthetic (bool): Whether to merge synthetic data.

    Returns:
        pd.DataFrame: Processed dataset ready for feature engineering.
    """
    df_train = load_csv(train_file)
    df = df_train.copy()

    if include_synthetic and synthetic_file.exists():
        df_synth = load_csv(synthetic_file)
        df = pd.concat([df_train, df_synth], ignore_index=True)

    df = preprocess(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    df = load_and_preprocess()
    print(f"[INFO] Processed dataset ready: {df.shape[0]} rows")
