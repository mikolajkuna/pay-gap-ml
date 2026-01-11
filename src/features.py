"""
Feature engineering module for PayGap-ML project.

Transforms the processed dataset into model-ready features (X) and target (y).
Handles categorical variables, missing values, and selects relevant columns.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN

def create_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Transform raw processed dataset into features (X) and target (y).

    Args:
        df (pd.DataFrame): Processed dataset.

    Returns:
        X (pd.DataFrame): Feature matrix ready for ML models.
        y (pd.Series): Target vector.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe")

    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"[INFO] Categorical columns: {categorical_cols}")
    print(f"[INFO] Numerical columns: {numerical_cols}")

    # Define preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    # Fit-transform features
    X_transformed = preprocess.fit_transform(X)
    # Convert back to DataFrame for convenience
    feature_names = (
        preprocess.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
        + numerical_cols
    )
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    print(f"[INFO] Features shape after encoding: {X_df.shape}")

    return X_df, y


if __name__ == "__main__":
    from pathlib import Path
    from src.dataset import build_dataset
    from src.config import PROCESSED_DATA_DIR

    # Build dataset
    df = build_dataset()
    
    # Create features
    X, y = create_features(df)

    print(f"[INFO] Feature matrix X shape: {X.shape}")
    print(f"[INFO] Target vector y shape: {y.shape}")
