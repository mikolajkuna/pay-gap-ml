import argparse
from predict_xgb import predict_xgb
from predict_bayes import predict_bayes
from predict_bart import predict_bart
from predict_tabpfn import predict_tabpfn
from predict_autogluon import predict_autogluon
from predict_lr import predict_lr
from predict_rf import predict_rf

MODELS = {
    "xgb": predict_xgb,
    "bayes": predict_bayes,
    "bart": predict_bart,
    "tabpfn": predict_tabpfn,
    "autogluon": predict_autogluon,
    "lr": predict_lr,
    "rf": predict_rf
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict income with selected ML model")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    args = parser.parse_args()

    print(f"Running predictions for model: {args.model}")
    MODELS[args.model]()
