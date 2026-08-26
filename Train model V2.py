"""
Reshape features_dataset_v2.csv into per-player rows, then retrain
the models with the expanded feature set (phases, opening family,
time usage) and compare against the previous baseline.

Usage:
    python train_model_v2.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_CSV = "features_dataset_v2.csv"
OUTPUT_CSV = "player_features_v2.csv"

NUMERIC_FEATURES = [
    "avg_cpl", "blunders", "mistakes", "inaccuracies", "num_moves",
    "opening_cpl", "middlegame_cpl", "endgame_cpl",
    "avg_time_per_move", "time_std",
]
TARGET_COLUMN = "elo"


def reshape_to_per_player(df):
    def side_frame(color):
        return pd.DataFrame({
            "elo": df[f"{color}_elo"],
            "avg_cpl": df[f"{color}_avg_cpl"],
            "blunders": df[f"{color}_blunders"],
            "mistakes": df[f"{color}_mistakes"],
            "inaccuracies": df[f"{color}_inaccuracies"],
            "opening_cpl": df[f"{color}_opening_cpl"],
            "middlegame_cpl": df[f"{color}_middlegame_cpl"],
            "endgame_cpl": df[f"{color}_endgame_cpl"],
            "avg_time_per_move": df[f"{color}_avg_time_per_move"],
            "time_std": df[f"{color}_time_std"],
            "eco_family": df["eco_family"],
            "num_moves": df["num_moves"],
        })

    return pd.concat([side_frame("white"), side_frame("black")], ignore_index=True)


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:<20} MAE: {mae:>7.1f}   RMSE: {rmse:>7.1f}")


def main():
    games_df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(games_df):,} games.")

    players_df = reshape_to_per_player(games_df)

    # Numeric columns may contain blanks (e.g. no endgame reached) -- coerce
    # to numeric and fill missing values with the column median.
    for col in NUMERIC_FEATURES:
        players_df[col] = pd.to_numeric(players_df[col], errors="coerce")
        players_df[col] = players_df[col].fillna(players_df[col].median())

    missing_time_pct = players_df["avg_time_per_move"].isna().mean() * 100
    if missing_time_pct > 50:
        print("Warning: most rows are missing time data -- clock annotations "
              "may not be present in your PGN sample.")

    # One-hot encode opening family (A-E)
    eco_dummies = pd.get_dummies(players_df["eco_family"], prefix="eco")
    players_df = pd.concat([players_df, eco_dummies], axis=1)

    feature_columns = NUMERIC_FEATURES + list(eco_dummies.columns)
    players_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Reshaped into {len(players_df):,} rows. Saved to {OUTPUT_CSV}.\n")

    X = players_df[feature_columns]
    y = players_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows.\n")
    print("--- Results (lower is better) ---")

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    evaluate("Naive baseline", y_test, baseline_pred)

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    evaluate("Linear regression", y_test, lin_model.predict(X_test))

    rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate("Random forest", y_test, rf_model.predict(X_test))

    print("\n--- Feature importance (random forest) ---")
    importances = pd.Series(rf_model.feature_importances_, index=feature_columns)
    print(importances.sort_values(ascending=False))

    print("\n(Compare against the v1 model: Linear MAE 222.2, Random forest MAE 227.9)")


if __name__ == "__main__":
    main()
