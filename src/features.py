"""
features.py

Feature engineering module for PayGap-ML project.
Transforms raw/preprocessed dataset into features ready for ML models.
"""

import pandas as pd
import numpy as np

def preprocess_gender(df: pd.DataFrame, column: str = "gender") -> pd.DataFrame:
    """
    Map gender values to numeric (1 = male, 0 = female).
    """
    gender_map = {"M": 1, "Male": 1, "m": 1, "F": 0, "Female": 0, "f": 0}
    df[column] = df[column].map(gender_map)
    return df

def preprocess_numeric(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Convert specified columns to numeric and handle invalid entries.
    """
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def filter_valid_rows(df: pd.DataFrame, income_col: str = "income", numeric_cols: list = None) -> pd.DataFrame:
    """
    Keep only rows with valid numeric values and realistic income.
    """
    if numeric_cols is None:
        numeric_cols = ["age", "experience_years", "absence", "child",
                        "education_level", "job_level", "distance_from_home", income_col]
    df = df[(df[income_col] >= 1276) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features if needed. For now, no extra derived features.
    Placeholder for future feature engineering.
    """

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline: map gender, convert numerics, filter invalid rows,
    add derived features.
    """
    df = df.copy()
    

    df = preprocess_gender(df)
    
 
    numeric_cols = ["age", "experience_years", "absence", "child",
                    "education_level", "job_level", "distance_from_home", "income"]
    
 
    df = preprocess_numeric(df, numeric_cols)
    
   
    df = filter_valid_rows(df, numeric_cols=numeric_cols)
    
    df = add_derived_features(df)
    
    return df

def get_feature_columns() -> list:
    """
    Returns the list of features to use for modeling.
    """
    return ["age", "gender", "education_level", "job_level",
            "experience_years", "distance_from_home", "absence", "child"]
