"""
Feature engineering module for PayGap-ML project.

Transforms raw/preprocessed dataset into features ready for ML models.
"""

from typing import List, Optional
import pandas as pd


def preprocess_gender(df: pd.DataFrame, column: str = "gender") -> pd.DataFrame:
    """
    Map gender values to numeric (1 = male, 0 = female).

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name containing gender.

    Returns:
        pd.DataFrame: Dataframe with numeric gender.
    """
    df = df.copy()
    gender_map = {"M": 1, "Male": 1, "m": 1, "F": 0, "Female": 0, "f": 0}
    df[column] = df[column].map(gender_map)
    return df


def preprocess_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric and handle invalid entries.

    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_cols (List[str]): List of numeric column names.

    Returns:
        pd.DataFrame: Dataframe with numeric columns.
    """
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_valid_rows(
    df: pd.DataFrame,
    income_col: str = "income",
    numeric_cols: Optional[List[str]] = None,
    min_income: float = 1276
) -> pd.DataFrame:
    """
    Keep only rows with valid numeric values and realistic income.

    Args:
        df (pd.DataFrame): Input dataframe.
        income_col (str): Column name for income.
        numeric_cols (List[str], optional): Columns to validate as numeric.
        min_income (float): Minimum realistic income.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = ["age", "experience_years", "absence", "child",
                        "education_level", "job_level", "distance_from_home", income_col]
    df = df[(df[income_col] >= min_income) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features if needed. Placeholder for future feature engineering.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with derived features.
    """
    return df.copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    - map gender
    - convert numeric columns
    - filter invalid rows
    - add derived features

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe ready for modeling.
    """
    df = df.copy()
    df = preprocess_gender(df)
    numeric_cols = ["age", "experience_years", "absence", "child",
                    "education_level", "job_level", "distance_from_home", "income"]
    df = preprocess_numeric(df, numeric_cols)
    df = filter_valid_rows(df, numeric_cols=numeric_cols)
    df = add_derived_features(df)
    return df


def get_feature_columns() -> List[str]:
    """
    Returns the list of features to use for modeling.

    Returns:
        List[str]: Names of feature columns.
    """
    return ["age", "gender", "education_level", "job_level",
            "experience_years", "distance_from_home", "absence", "child"]

FEATURES = get_feature_columns()
