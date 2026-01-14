from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from autogluon.tabular import TabularPredictor
from ISLP.bart import BART
from tabpfn import TabPFNRegressor

from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================
# Configuration
# =========================

DATA_BASE_URL: str = (
    "https://raw.githubusercontent.com/"
    "mikolajkuna/pay-gap-ml/main/data/processed"
)

TRAIN_DATA_URL: str = f"{DATA_BASE_URL}/salary_data_train.csv"
TEST_DATA_URL: str = f"{DATA_BASE_URL}/salary_data_synthetic.csv"

FEATURE_COLUMNS: list[str] = [
    "age",
    "gender",
    "education_level",
    "job_level",
    "experience_years",
    "child",
    "distance_from_home",
    "absence",
]

SHAP_FEATURES: list[str] = [
    "gender",
    "child",
    "education_level",
    "job_level",
]


# =========================
# Data
# =========================

class DatasetLoader:
    def load(self, path: str) -> pd.DataFrame:
        sample: pd.DataFrame = pd.read_csv(path, nrows=1)
        separator: str = ";" if ";" in sample.columns[0] else ","
        return pd.read_csv(path, sep=separator)


class SalaryPreprocessor:
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        processed: pd.DataFrame = data.copy()

        processed["gender"] = processed["gender"].replace(
            {
                "M": 1,
                "Male": 1,
                "m": 1,
                "F": 0,
                "Female": 0,
                "f": 0,
            }
        )

        education_mapping: dict[object, int] = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            "Bachelor's": 1,
            "Master's": 2,
            "PhD": 3,
            "Other": 4,
        }
        processed["education_level"] = processed["education_level"].replace(
            education_mapping
        )

        numeric_columns: list[str] = [
            "age",
            "experience_years",
            "absence",
            "child",
            "education_level",
            "job_level",
            "distance_from_home",
            "income",
        ]

        for column in numeric_columns:
            processed[column] = pd.to_numeric(processed[column], errors="coerce")

        processed = processed[
            (processed["income"] >= 1276)
            & (processed[numeric_columns].ge(0).all(axis=1))
        ]

        return processed.dropna()


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    target: pd.Series


class DatasetBuilder:
    def build(self, data: pd.DataFrame) -> Dataset:
        return Dataset(
            features=data[FEATURE_COLUMNS].astype(np.float32),
            target=data["income"].astype(np.float32),
        )


# =========================
# Models
# =========================

class TrainableModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class LinearRegressionModel:
    def __init__(self) -> None:
        self._model: RegressorMixin = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class BayesianRidgeModel:
    def __init__(self) -> None:
        self._model: RegressorMixin = BayesianRidge()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class RandomForestModel:
    def __init__(self) -> None:
        self._model: RegressorMixin = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class XGBoostModel:
    def __init__(self) -> None:
        self._model: xgb.XGBRegressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class BartModel:
    def __init__(self) -> None:
        self._model: BART = BART(
            num_trees=150,
            num_particles=8,
            max_stages=4000,
            burnin=300,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class TabPFNModel:
    def __init__(self) -> None:
        self._model: TabPFNRegressor = TabPFNRegressor(random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class AutoGluonModel:
    def __init__(self) -> None:
        self._predictor: TabularPredictor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        training_data: pd.DataFrame = X.copy()
        training_data["income"] = y.values

        self._predictor = TabularPredictor(
            label="income",
            verbosity=0,
        ).fit(training_data)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError("Model has not been fitted")

        return self._predictor.predict(X).to_numpy()


# =========================
# Evaluation
# =========================

@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    cv: float
    mean_true: float
    mean_predicted: float


class RegressionEvaluator:
    def evaluate(
        self,
        model: TrainableModel,
        dataset: Dataset,
    ) -> RegressionMetrics:
        predictions: np.ndarray = model.predict(dataset.features)

        mae: float = mean_absolute_error(dataset.target, predictions)
        cv: float = np.std(dataset.target - predictions) / np.mean(dataset.target) * 100

        return RegressionMetrics(
            mae=mae,
            cv=cv,
            mean_true=float(dataset.target.mean()),
            mean_predicted=float(predictions.mean()),
        )


# =========================
# Gender Pay Gap
# =========================

@dataclass(frozen=True)
class GenderGapResult:
    mean_gap_pln: float
    mean_gap_pct: float
    median_gap_pln: float
    median_gap_pct: float


class GenderGapAnalyzer:
    def analyze(
        self,
        model: TrainableModel,
        features: pd.DataFrame,
    ) -> GenderGapResult:
        counterfactual: pd.DataFrame = features.copy()
        counterfactual["gender"] = 1 - counterfactual["gender"]

        factual: np.ndarray = model.predict(features)
        counterfactual_pred: np.ndarray = model.predict(counterfactual)

        gap_pln: np.ndarray = factual - counterfactual_pred
        gap_pct: np.ndarray = gap_pln / factual * 100

        return GenderGapResult(
            mean_gap_pln=float(np.mean(gap_pln)),
            mean_gap_pct=float(np.mean(gap_pct)),
            median_gap_pln=float(np.median(gap_pln)),
            median_gap_pct=float(np.median(gap_pct)),
        )


# =========================
# Explainability
# =========================

class ShapExplainer:
    def explain(
        self,
        model: TrainableModel,
        features: pd.DataFrame,
        sample_size: int = 150,
    ) -> None:
        sample: pd.DataFrame = features.sample(
            n=sample_size,
            random_state=42,
        )
        reduced: pd.DataFrame = sample[SHAP_FEATURES]

        def prediction_proxy(data: pd.DataFrame) -> np.ndarray:
            reconstructed: pd.DataFrame = sample.copy()
            reconstructed[SHAP_FEATURES] = data.values
            return model.predict(reconstructed)

        explainer = shap.Explainer(prediction_proxy, reduced)
        shap_values = explainer(reduced)

        shap.summary_plot(shap_values, reduced, plot_type="bar")
        shap.summary_plot(shap_values, reduced)


# =========================
# Pipeline
# =========================

class TrainingPipeline:
    def __init__(self, model: TrainableModel) -> None:
        self._loader = DatasetLoader()
        self._preprocessor = SalaryPreprocessor()
        self._builder = DatasetBuilder()
        self._evaluator = RegressionEvaluator()
        self._gap_analyzer = GenderGapAnalyzer()
        self._explainer = ShapExplainer()
        self._model = model

    def run(self) -> None:
        train_raw = self._loader.load(TRAIN_DATA_URL)
        test_raw = self._loader.load(TEST_DATA_URL)

        train = self._builder.build(
            self._preprocessor.preprocess(train_raw)
        )
        test = self._builder.build(
            self._preprocessor.preprocess(test_raw)
        )

        self._model.fit(train.features, train.target)

        print(self._evaluator.evaluate(self._model, test))
        print(self._gap_analyzer.analyze(self._model, test.features))

        self._explainer.explain(self._model, test.features)


def main() -> None:
    pipeline = TrainingPipeline(
        model=AutoGluonModel(),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
