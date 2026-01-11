"""
Training module for PayGap-ML project.

Contains functions to train classical ML models (Linear Regression, Random Forest)
and placeholder functions for pre-trained tabular models (TabPFN, LimiX).
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# TODO: Uncomment and implement when TabPFN and LimiX are installed
# from src.modeling.tabpfn import TabPFNRegressor
# from src.modeling.limix import LimixRegressor

from src.config import PROCESSED_DATA_DIR

def load_data(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")
    return df


def train_classical_models(
    df: pd.DataFrame,
    target: str = "pay_gap"
) -> Dict[str, Pipeline]:
    """
    Train classical ML models using a preprocessing pipeline.

    Args:
        df (pd.DataFrame): Input dataset.
        target (str): Name of the target variable.

    Returns:
        Dict[str, Pipeline]: Trained models keyed by model name.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Define models
    models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained_models: Dict[str, Pipeline] = {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        print(f"[RESULT] {name} | RMSE: {rmse:.3f}, R2: {r2:.3f}")
        trained_models[name] = pipe

    return trained_models


def train_tabular_models(
    df: pd.DataFrame,
    target: str = "pay_gap"
) -> Dict[str, Any]:
    """
    Placeholder function for pre-trained tabular models (TabPFN, LimiX).

    Args:
        df (pd.DataFrame): Input dataset.
        target (str): Name of the target variable.

    Returns:
        Dict[str, Any]: Dictionary of trained models.
    """
    # TODO: Implement training for TabPFN and LimiX
    print("[INFO] Pre-trained tabular models training not implemented yet")
    return {}


def train_all(path: Path, target: str = "pay_gap") -> Dict[str, Any]:
    """
    Load dataset and train all models (classical + pre-trained tabular).

    Args:
        path (Path): Path to the processed CSV dataset.
        target (str): Name of the target variable.

    Returns:
        Dict[str, Any]: Dictionary of all trained models.
    """
    df = load_data(path)

    print("\n[INFO] Training classical ML models")
    classical_models = train_classical_models(df, target)

    print("\n[INFO] Training pre-trained tabular models")
    tabular_models = train_tabular_models(df, target)

    all_models = {**classical_models, **tabular_models}
    print(f"\n[INFO] Training complete. Total models trained: {len(all_models)}")
    return all_models


if __name__ == "__main__":
    processed_file = PROCESSED_DATA_DIR / "paygap.csv"
    trained_models = train_all(processed_file)
