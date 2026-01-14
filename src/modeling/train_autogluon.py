from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

class AutoGluonTrainer:
    """
    AutoGluon trainer class for tabular regression.
    Trains model on processed salary dataset.
    """

    def __init__(
        self,
        data_path: Path,
        label_column: str,
    ) -> None:
        self.data_path = data_path
        self.label_column = label_column
        self.predictor: TabularPredictor | None = None
        self.training_data: pd.DataFrame | None = None

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path, sep=";")
        df_clean = df.dropna(subset=[self.label_column])
        self.training_data = df_clean

    def train(self) -> TabularPredictor:
        """
        Train AutoGluon model on the dataset.
        The TabularPredictor object is returned.
        """
        self.predictor = TabularPredictor(
            label=self.label_column,
            verbosity=0,
        ).fit(self.training_data)
        return self.predictor

    def evaluate(self) -> dict[str, float]:
        """
        Returns training evaluation metrics (using same training data).
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been trained yet.")
        performance = self.predictor.evaluate(self.training_data)
        return performance


def main() -> None:
    train_path = Path("data/processed/salary_data_train.csv")
    label = "income"

    trainer = AutoGluonTrainer(data_path=train_path, label_column=label)
    trainer.load_data()
    predictor = trainer.train()
    metrics = trainer.evaluate()

    print("\n--- AutoGluon train metrics ---")
    print(metrics)


if __name__ == "__main__":
    main()
