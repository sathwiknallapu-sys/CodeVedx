"""
visualize.py
-------------
Generates exploratory data analysis (EDA) charts for the report and
README. Saves all plots as PNG files into the docs/ folder.

Charts produced:
    1. Distribution of final marks
    2. Performance category counts
    3. Correlation heatmap
    4. Study hours vs final marks scatter plot
    5. Feature importance (from Random Forest, for interpretability)
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

import config
from feature_engineering import engineer_features

sns.set_theme(style="whitegrid")
OUTPUT_DIR = os.path.join(config.BASE_DIR, "docs", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_marks_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["final_marks"], bins=20, kde=True, color="#2E86AB")
    plt.title("Distribution of Final Marks")
    plt.xlabel("Final Marks")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "marks_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_category_counts(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    order = ["At Risk", "Average", "Good", "Excellent"]
    sns.countplot(
        data=df, x="performance_category", order=order,
        hue="performance_category", palette="viridis", legend=False,
    )
    plt.title("Student Count by Performance Category")
    plt.xlabel("Performance Category")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "category_counts.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_study_vs_marks(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="study_hours_per_day",
        y="final_marks",
        hue="performance_category",
        hue_order=["At Risk", "Average", "Good", "Excellent"],
        palette="viridis",
        alpha=0.7,
    )
    plt.title("Study Hours vs Final Marks")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Final Marks")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "study_hours_vs_marks.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance():
    """Loads the trained Random Forest (if available) purely to
    visualize feature importance for interpretability, even if the
    deployed model is Linear Regression.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df = engineer_features(df)
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_REGRESSION]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_scaled, y)

    importance_df = pd.DataFrame({
        "feature": config.ALL_FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#3CAEA3")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def generate_all_charts():
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df = engineer_features(df)

    plot_marks_distribution(df)
    plot_category_counts(df)
    plot_correlation_heatmap(df)
    plot_study_vs_marks(df)
    plot_feature_importance()
    print("\nAll charts generated successfully.")


if __name__ == "__main__":
    generate_all_charts()
