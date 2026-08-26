"""
Reshape features_dataset.csv (one row per game, both players side by side)
into player_features.csv (one row per player-performance), and check
whether rating actually correlates with playing accuracy.

Usage:
    python explore_features.py
"""

import pandas as pd

INPUT_CSV = "features_dataset.csv"
OUTPUT_CSV = "player_features.csv"


def reshape_to_per_player(df):
    """Turn each game row into two rows: one for White's performance,
    one for Black's."""
    white_rows = pd.DataFrame({
        "elo": df["white_elo"],
        "avg_cpl": df["white_avg_cpl"],
        "blunders": df["white_blunders"],
        "mistakes": df["white_mistakes"],
        "inaccuracies": df["white_inaccuracies"],
        "color": "white",
        "opening": df["opening"],
        "time_control": df["time_control"],
        "num_moves": df["num_moves"],
    })

    black_rows = pd.DataFrame({
        "elo": df["black_elo"],
        "avg_cpl": df["black_avg_cpl"],
        "blunders": df["black_blunders"],
        "mistakes": df["black_mistakes"],
        "inaccuracies": df["black_inaccuracies"],
        "color": "black",
        "opening": df["opening"],
        "time_control": df["time_control"],
        "num_moves": df["num_moves"],
    })

    return pd.concat([white_rows, black_rows], ignore_index=True)


def main():
    games_df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(games_df):,} games.\n")

    players_df = reshape_to_per_player(games_df)
    players_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Reshaped into {len(players_df):,} player-performance rows.")
    print(f"Saved to {OUTPUT_CSV}\n")

    print("--- The key question: does rating correlate with accuracy? ---")
    correlation = players_df[["elo", "avg_cpl", "blunders", "mistakes", "inaccuracies"]].corr()["elo"]
    print(correlation, "\n")
    print("(We want elo-vs-avg_cpl to be clearly NEGATIVE: higher rating, lower")
    print("centipawn loss. Same for blunders/mistakes/inaccuracies.)\n")

    print("--- Average centipawn loss by rating bracket ---")
    bins = [1000, 1200, 1400, 1600, 1800, 2000, 2200]
    players_df["rating_bracket"] = pd.cut(players_df["elo"], bins=bins)
    bracket_summary = players_df.groupby("rating_bracket", observed=True)["avg_cpl"].agg(["mean", "count"])
    print(bracket_summary)


if __name__ == "__main__":
    main()
