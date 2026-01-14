# src/mikolajkuna/plots.py

"""
Module for visualizing PayGap-ML results.
"""

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap

sns.set(style="whitegrid")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predicted vs True Income") -> None:
    """
    Scatter plot of predicted vs true values.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("True Income")
    plt.ylabel("Predicted Income")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_gap_distribution(gap_pln: np.ndarray, title: str = "Gender Pay Gap Distribution") -> None:
    """
    Histogram of counterfactual gender pay gap in PLN.

    Args:
        gap_pln (np.ndarray): Array of income differences (male - female).
        title (str): Plot title.
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


def plot_shap_summary(shap_values: shap.Explanation, X: pd.DataFrame, plot_type: str = "bar") -> None:
    """
    SHAP summary plot.

    Args:
        shap_values (shap.Explanation): SHAP values from a model.
        X (pd.DataFrame): Feature data used for SHAP calculation.
        plot_type (str): Type of SHAP summary plot ("bar" or "dot").
    """
    shap.summary_plot(shap_values, X, plot_type=plot_type)


def plot_boxplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, title: Optional[str] = None) -> None:
    """
    Generic boxplot for visualizing distributions.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): Column for x-axis.
        y (str): Column for y-axis.
        hue (str, optional): Column for color grouping.
        title (str, optional): Plot title. If None, a default title is used.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.title(title if title else f"{y} by {x}")
    plt.tight_layout()
    plt.show()
