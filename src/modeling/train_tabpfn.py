import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error

class TabPFNTrainer:
    """
    Trainer for TabPFN regression model.
    Automatically asks for HuggingFace token if not set in environment.
    """
    def __init__(self, data_path: str, feature_columns: list, target_column: str = "income") -> None:
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model: TabPFNRegressor | None = None
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None

        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
            token = input("Please enter your HuggingFace token (HF_TOKEN): ").strip()
            os.environ["HF_TOKEN"] = token

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path, sep=";")
        df = df.dropna(subset=self.feature_columns + [self.target_column])
        self.X_train = df[self.feature_columns].astype(np.float32)
        self.y_train = df[self.target_column].astype(np.float32)

    def train(self, num_trees: int = 150, num_particles: int = 8) -> None:
        self.model = TabPFNRegressor(
            num_trees=num_trees,
            num_particles=num_particles,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> None:
        preds = self.model.predict(self.X_train)
        mae = mean_absolute_error(self.y_train, preds)
        cv = np.std(self.y_train - preds) / np.mean(self.y_train) * 100
        print("\n--- Training evaluation ---")
        print(f"MAE: {mae:,.0f} PLN")
        print(f"CV: {cv:.2f}%")
        print(f"Mean income (true): {self.y_train.mean():,.0f} PLN")
        print(f"Mean income (pred): {preds.mean():,.0f} PLN")

def main() -> None:
    data_path = "data/processed/salary_data_train.csv"
    features = [
        "age", "gender", "education_level", "job_level",
        "experience_years", "distance_from_home", "absence", "child"
    ]

    trainer = TabPFNTrainer(data_path=data_path, feature_columns=features)
    trainer.load_data()
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
