"""
plots.py

Module for visualizing PayGap-ML results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title="Predicted vs True Income"):
    """
    Scatter plot of predicted vs true values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("True Income")
    plt.ylabel("Predicted Income")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_gap_distribution(gap_pln: np.ndarray, title="Gender Pay Gap Distribution"):
    """
    Histogram of contractfactual gap in PLN.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(gap_pln, bins=30, kde=True, color="skyblue")
    plt.axvline(np.mean(gap_pln), color="red", linestyle="--", label=f"Mean: {np.mean(gap_pln):.0f} PLN")
    plt.axvline(np.median(gap_pln), color="green", linestyle="--", label=f"Median: {np.median(gap_pln):.0f} PLN")
    plt.xlabel("Gap (PLN)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_shap_summary(shap_values, X: pd.DataFrame, plot_type="bar"):
    """
    SHAP summary plot.
    """
    import shap

    shap.summary_plot(shap_values, X, plot_type=plot_type)


def plot_boxplot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None):
    """
    Generic boxplot.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.title(title if title else f"{y} by {x}")
    plt.tight_layout()
    plt.show()
