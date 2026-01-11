"""
Visualization module for PayGap-ML project.

Contains functions for plotting distributions, correlations, and feature importance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import ROOT_DIR

# Default folder for saving figures
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df, target_column="pay_gap", save: bool = True):
    """
    Plot histogram and boxplot for the target variable.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target_column], kde=True, bins=30)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    if save:
        plt.savefig(FIGURES_DIR / f"{target_column}_histogram.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[target_column])
    plt.title(f"Boxplot of {target_column}")
    if save:
        plt.savefig(FIGURES_DIR / f"{target_column}_boxplot.png", dpi=300)
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=20, save: bool = True):
    """
    Plot top N feature importances.
    
    Args:
        feature_names (list[str]): Names of features.
        importances (array-like): Importance values (from model.feature_importances_).
        top_n (int): Number of top features to plot.
    """
    # Create DataFrame for plotting
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    if save:
        plt.savefig(FIGURES_DIR / f"top_{top_n}_feature_importances.png", dpi=300)
    plt.show()
