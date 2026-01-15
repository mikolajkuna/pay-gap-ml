from tabpfn import TabPFNRegressor
from src.config import TRAIN_CSV, FEATURES
from src.features.feature_engineering import build_features
import pandas as pd
import os

class TabPFNTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.y_train = None

    def load_data(self, csv_path):
        """
        Load data and preprocess it.
        """
        df = pd.read_csv(csv_path)
        df = build_features(df)
        self.X_train = df[FEATURES]
        self.y_train = df["income"]

    def train(self):
        """
        Trains the TabPFN model and optionally asks for Hugging Face token if not set.
        """
        # Zapytaj o token jeśli nie jest ustawiony w środowisku
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            hf_token = input("Please enter your Hugging Face token: ")
            os.environ["HF_TOKEN"] = hf_token
        
        # Inicjalizowanie modelu
        self.model = TabPFNRegressor(device="cpu")
        self.model.fit(self.X_train, self.y_train)
        
        # Save the trained model (optional)
        print("Model training complete.")

    def evaluate_training(self):
        """
        Evaluate the model's performance on training data (optional).
        """
        if self.model:
            predictions = self.model.predict(self.X_train)
            # Evaluate the model (for instance, R² score or mean squared error)
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(self.y_train, predictions)
            r2 = r2_score(self.y_train, predictions)
            print(f"MSE: {mse}, R²: {r2}")
