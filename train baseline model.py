"""
Train baseline models to predict a player's Elo from their playing-style
features, and compare against a naive baseline (always guess the average).

Usage:
    python train_baseline_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_CSV = "player_features.csv"

FEATURE_COLUMNS = [
    "avg_cpl", "blunders", "mistakes", "inaccuracies", "num_moves",
]
TARGET_COLUMN = "elo"


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:<20} MAE: {mae:>7.1f}   RMSE: {rmse:>7.1f}")


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows.\n")
    print("--- Results (lower is better) ---")

    # Baseline: always predict the average rating in the training set
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    evaluate("Naive baseline", y_test, baseline_pred)

    # Linear regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    evaluate("Linear regression", y_test, lin_pred)

    # Random forest
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    evaluate("Random forest", y_test, rf_pred)

    print("\n--- Feature importance (random forest) ---")
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False))


if __name__ == "__main__":
    main()
