"""
Load sample_games.pgn into a pandas dataframe and look at basic stats.

Usage:
    python explore_games.py
"""

import chess.pgn
import pandas as pd

INPUT_FILE = "sample_games.pgn"
OUTPUT_CSV = "games_summary.csv"


def extract_row(game):
    headers = game.headers
    moves = list(game.mainline_moves())

    return {
        "white_elo": int(headers.get("WhiteElo", 0)),
        "black_elo": int(headers.get("BlackElo", 0)),
        "eco": headers.get("ECO", "unknown"),
        "opening": headers.get("Opening", "unknown"),
        "time_control": headers.get("TimeControl", "unknown"),
        "event": headers.get("Event", "unknown"),
        "result": headers.get("Result", "unknown"),
        "num_moves": len(moves),
    }


def load_games(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            rows.append(extract_row(game))
    return pd.DataFrame(rows)


def main():
    df = load_games(INPUT_FILE)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Loaded {len(df):,} games into a dataframe.\n")

    print("--- Rating stats ---")
    print(df[["white_elo", "black_elo"]].describe(), "\n")

    print("--- Game length (moves) ---")
    print(df["num_moves"].describe(), "\n")

    print("--- Top 10 openings ---")
    print(df["opening"].value_counts().head(10), "\n")

    print("--- Result breakdown ---")
    print(df["result"].value_counts(), "\n")

    print(f"Saved full table to {OUTPUT_CSV} — open it in Excel or pandas anytime.")


if __name__ == "__main__":
    main()
