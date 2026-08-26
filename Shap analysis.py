"""
Interpretability step: explain the random forest's predictions with SHAP,
and look at where the model's predictions are most wrong.

Produces two saved images:
  - shap_bar.png       (which features matter most, on average)
  - shap_beeswarm.png  (how each feature pushes predictions up or down)

Usage:
    python shap_analysis.py
"""

import matplotlib
matplotlib.use("Agg")  # avoids needing a pop-up window; we save plots to files instead
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

INPUT_CSV = "player_features_v2.csv"

NUMERIC_FEATURES = [
    "avg_cpl", "blunders", "mistakes", "inaccuracies", "num_moves",
    "opening_cpl", "middlegame_cpl", "endgame_cpl",
    "avg_time_per_move", "time_std",
]
TARGET_COLUMN = "elo"


def main():
    df = pd.read_csv(INPUT_CSV)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # player_features_v2.csv may already contain eco_A..eco_E from the
    # previous script -- drop those (but NOT eco_family itself) so we
    # don't create duplicates.
    existing_eco_cols = [c for c in df.columns if c.startswith("eco_") and c != "eco_family"]
    df = df.drop(columns=existing_eco_cols)

    eco_dummies = pd.get_dummies(df["eco_family"], prefix="eco")
    feature_columns = NUMERIC_FEATURES + list(eco_dummies.columns)
    df = pd.concat([df, eco_dummies], axis=1)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("Computing SHAP values (this takes a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Plot 1: overall feature importance, SHAP style ---
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150)
    plt.close()
    print("Saved shap_bar.png")

    # --- Plot 2: how each feature pushes predictions up/down, per example ---
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)
    plt.close()
    print("Saved shap_beeswarm.png")

    # --- Error analysis: where is the model most wrong? ---
    predictions = model.predict(X_test)
    errors = pd.DataFrame({
        "actual_elo": y_test.values,
        "predicted_elo": predictions,
        "abs_error": np.abs(y_test.values - predictions),
    }, index=X_test.index)

    # bring back the original feature values for context
    errors = errors.join(X_test)

    print("\n--- 10 worst predictions ---")
    worst = errors.sort_values("abs_error", ascending=False).head(10)
    print(worst[["actual_elo", "predicted_elo", "abs_error", "avg_cpl", "num_moves"]])

    print("\n--- 10 best predictions ---")
    best = errors.sort_values("abs_error", ascending=True).head(10)
    print(best[["actual_elo", "predicted_elo", "abs_error", "avg_cpl", "num_moves"]])

    print(f"\nMean absolute error overall: {errors['abs_error'].mean():.1f}")
    print(f"Median absolute error: {errors['abs_error'].median():.1f}")


if __name__ == "__main__":
    main()
