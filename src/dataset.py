"""
Dataset module for PayGap-ML project.

Loads raw CSV files, optionally merges synthetic data, 
and saves processed dataset ready for feature engineering and modeling.
"""

from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def build_dataset(
    train_file: Path = RAW_DATA_DIR / "salary_data_train.csv",
    synthetic_file: Path = RAW_DATA_DIR / "salary_data_synthetic.csv",
    output_file: Path = PROCESSED_DATA_DIR / "paygap.csv",
    include_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load training and optional synthetic CSVs, merge, and save processed dataset.

    Args:
        train_file (Path): Path to main training CSV.
        synthetic_file (Path): Path to synthetic CSV.
        output_file (Path): Path to save processed dataset.
        include_synthetic (bool): Whether to merge synthetic data.

    Returns:
        pd.DataFrame: Processed dataset ready for feature engineering.
    """
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    # Load training data
    df_train = pd.read_csv(train_file)
    print(f"[INFO] Loaded training data: {train_file} ({df_train.shape[0]} rows)")

    # Optionally load and merge synthetic data
    if include_synthetic and synthetic_file.exists():
        df_synth = pd.read_csv(synthetic_file)
        print(f"[INFO] Loaded synthetic data: {synthetic_file} ({df_synth.shape[0]} rows)")
        df = pd.concat([df_train, df_synth], ignore_index=True)
        print(f"[INFO] Combined dataset shape: {df.shape}")
    else:
        df = df_train

    # OPTIONAL: preprocessing steps can be added here
    # e.g., renaming columns, filling missing values, converting types
    # df["gender"] = df["gender"].astype(str)
    # df.fillna(method="ffill", inplace=True)

    # Ensure processed directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(output_file, index=False)
    print(f"[INFO] Processed dataset saved to {output_file}")

    return df


if __name__ == "__main__":
    # Domyślnie scalamy train + synthetic
    build_dataset()
