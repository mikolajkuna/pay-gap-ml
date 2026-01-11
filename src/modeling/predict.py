"""
Prediction module for PayGap-ML project.

Contains functions to generate predictions from trained models:
- classical ML models (Linear Regression, Random Forest)
- placeholder for pre-trained tabular models (TabPFN, LimiX)
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

# TODO: Uncomment and implement when TabPFN and LimiX are installed
# from src.modeling.tabpfn import TabPFNRegressor
# from src.modeling.limix import LimixRegressor

from src.config import PROCESSED_DATA_DIR, MODELS_DIR


def load_models(models_dir: Path) -> Dict[str, Any]:
    """
    Load trained models from disk.

    Args:
        models_dir (Path): Directory where models are saved.

    Returns:
        Dict[str, Any]: Dictionary of models keyed by name.
    """
    models: Dict[str, Any] = {}
    if not models_dir.exists():
        print(f"[WARN] Models directory {models_dir} does not exist.")
        return models

    for model_file in models_dir.glob("*.joblib"):
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)
        print(f"[INFO] Loaded model: {model_name}")

    return models


def predict(models: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for all loaded models.

    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        df (pd.DataFrame): New data to predict on.

    Returns:
        pd.DataFrame: Predictions from all models as columns.
    """
    if not models:
        print("[WARN] No models provided for prediction.")
        return pd.DataFrame()

    predictions = pd.DataFrame(index=df.index)

    for name, model in models.items():
        try:
            preds = model.predict(df)
            predictions[name] = preds
            print(f"[INFO] Predictions generated: {name}")
        except Exception as e:
            print(f"[ERROR] Could not predict with {name}: {e}")

    return predictions


if __name__ == "__main__":
    # Domyślnie wczytujemy przetworzony plik
    processed_file = PROCESSED_DATA_DIR / "paygap.csv"
    df = pd.read_csv(processed_file)

    # Wczytaj wytrenowane modele
    trained_models = load_models(MODELS_DIR)

    # Generuj predykcje
    preds_df = predict(trained_models, df)

    # Zapisz predykcje do pliku
    output_file = MODELS_DIR / "predictions.csv"
    preds_df.to_csv(output_file, index=False)
    print(f"[INFO] Predictions saved to {output_file}")
