import argparse
from train_xgb import train_xgb
from train_bayes import train_bayes
from train_tabpfn import train_tabpfn
from train_autogluon import train_autogluon
from train_lr import train_lr
from train_rf import train_rf

MODELS = {
    "xgb": train_xgb,
    "bayes": train_bayes,
    "tabpfn": train_tabpfn,
    "autogluon": train_autogluon,
    "lr": train_lr,
    "rf": train_rf
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selected ML model")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    args = parser.parse_args()

    print(f"Training model: {args.model}")
    MODELS[args.model]()

