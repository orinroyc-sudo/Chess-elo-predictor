"""
Train the final model on the full dataset and save it (plus the metadata
needed to build matching features at inference time) to disk.

Usage:
    python train_and_save_model.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

INPUT_CSV = "player_features_v2.csv"
MODEL_FILE = "elo_model.joblib"

NUMERIC_FEATURES = [
    "avg_cpl", "blunders", "mistakes", "inaccuracies", "num_moves",
    "opening_cpl", "middlegame_cpl", "endgame_cpl",
    "avg_time_per_move", "time_std",
]
TARGET_COLUMN = "elo"


def main():
    df = pd.read_csv(INPUT_CSV)

    existing_eco_cols = [c for c in df.columns if c.startswith("eco_") and c != "eco_family"]
    df = df.drop(columns=existing_eco_cols)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save the medians used for imputation -- the live demo needs to use
    # these SAME values for any missing feature, not recompute its own.
    medians = df[NUMERIC_FEATURES].median().to_dict()
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(medians[col])

    eco_dummies = pd.get_dummies(df["eco_family"], prefix="eco")
    feature_columns = NUMERIC_FEATURES + list(eco_dummies.columns)
    df = pd.concat([df, eco_dummies], axis=1)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "medians": medians,
        "eco_categories": list(eco_dummies.columns),  # e.g. ['eco_A', ... 'eco_E']
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"Saved trained model and metadata to {MODEL_FILE}")


if __name__ == "__main__":
    main()
