import os
from src.modeling.predict_tabpfn import TabPFNPredictor
from src.config import MODEL_FILES


def predict_model():

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = input("Please enter your Hugging Face token: ")
        os.environ["HF_TOKEN"] = hf_token
    

    predictor = TabPFNPredictor()
    predictor.load_model(MODEL_FILES["tabpfn"])

    df_test = pd.read_csv("path_to_your_test_data.csv")  
    predictions = predictor.predict(df_test)

    print(f"Predictions: {predictions}")

predict_model()
