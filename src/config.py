"""
Configuration file for PayGap-ML project.

Contains paths to data directories, target column, and other global settings.
"""

from pathlib import Path

# Root directory of the project (repo root)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Target column for prediction
TARGET_COLUMN = "income"  

# Model output directory
MODEL_DIR = ROOT_DIR / "models"

# Optional: random seed for reproducibility
RANDOM_SEED = 42
