# src/dataset.py

from pathlib import Path
import pandas as pd
from src import features 

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe: map gender, convert numeric columns, handle invalid values.
    """

    df = features.preprocess_gender(df)


    numeric_cols = features.get_feature_columns() + ["income"]

    
    df = features.preprocess_numeric(df, numeric_cols)


    df = features.filter_valid_rows(df, numeric_cols=numeric_cols)

    return df
