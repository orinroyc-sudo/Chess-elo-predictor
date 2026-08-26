"""
Analyze a batch of games from sample_games.pgn with Stockfish and build
a feature table (one row per game) for later modeling.

Saves progress after every game, so it's safe to stop (Ctrl+C or close
the window) and resume later -- it picks up where it left off.

Usage:
    python batch_analyze.py
"""

import csv
import os
import time

import chess
import chess.engine
import chess.pgn

PGN_FILE = "sample_games.pgn"
OUTPUT_CSV = "features_dataset.csv"

STOCKFISH_PATH = (
    r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)

ANALYSIS_DEPTH = 12
GAMES_TO_PROCESS = 1200

MATE_SCORE = 1000

# Lichess-style thresholds for classifying a move by centipawn loss
INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD = 100
BLUNDER_THRESHOLD = 300

# Maps the label returned by classify_move() to the (irregularly plural)
# dictionary key used in the stats dict -- "inaccuracy" -> "inaccuracies"
# is NOT just "+ s", which was the bug in the previous version.
CLASS_TO_KEY = {
    "blunder": "blunders",
    "mistake": "mistakes",
    "inaccuracy": "inaccuracies",
}

FIELDNAMES = [
    "white_elo", "black_elo", "eco", "opening", "time_control", "result", "num_moves",
    "white_avg_cpl", "black_avg_cpl",
    "white_blunders", "white_mistakes", "white_inaccuracies",
    "black_blunders", "black_mistakes", "black_inaccuracies",
]


def score_to_cp(score, pov):
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        return MATE_SCORE if mate_in > 0 else -MATE_SCORE
    return relative.score()


def classify_move(cpl):
    if cpl >= BLUNDER_THRESHOLD:
        return "blunder"
    if cpl >= MISTAKE_THRESHOLD:
        return "mistake"
    if cpl >= INACCURACY_THRESHOLD:
        return "inaccuracy"
    return None


def analyze_game(game, engine):
    board = game.board()
    stats = {
        "white": {"cpl_total": 0, "moves": 0, "blunders": 0, "mistakes": 0, "inaccuracies": 0},
        "black": {"cpl_total": 0, "moves": 0, "blunders": 0, "mistakes": 0, "inaccuracies": 0},
    }

    for move in game.mainline_moves():
        mover = board.turn
        side_key = "white" if mover else "black"

        info_before = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        eval_before = score_to_cp(info_before["score"], mover)

        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        eval_after = score_to_cp(info_after["score"], mover)

        cpl = max(0, eval_before - eval_after)

        stats[side_key]["cpl_total"] += cpl
        stats[side_key]["moves"] += 1

        move_class = classify_move(cpl)
        if move_class:
            stats[side_key][CLASS_TO_KEY[move_class]] += 1

    return stats


def game_to_row(game, stats):
    headers = game.headers
    w, b = stats["white"], stats["black"]

    return {
        "white_elo": headers.get("WhiteElo", ""),
        "black_elo": headers.get("BlackElo", ""),
        "eco": headers.get("ECO", ""),
        "opening": headers.get("Opening", ""),
        "time_control": headers.get("TimeControl", ""),
        "result": headers.get("Result", ""),
        "num_moves": w["moves"] + b["moves"],
        "white_avg_cpl": round(w["cpl_total"] / max(1, w["moves"]), 1),
        "black_avg_cpl": round(b["cpl_total"] / max(1, b["moves"]), 1),
        "white_blunders": w["blunders"],
        "white_mistakes": w["mistakes"],
        "white_inaccuracies": w["inaccuracies"],
        "black_blunders": b["blunders"],
        "black_mistakes": b["mistakes"],
        "black_inaccuracies": b["inaccuracies"],
    }


def count_existing_rows():
    if not os.path.exists(OUTPUT_CSV):
        return 0
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header row


def main():
    already_done = count_existing_rows()
    if already_done > 0:
        print(f"Resuming: {already_done} games already processed, skipping those.")

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(PGN_FILE, encoding="utf-8") as pgn_f, \
         open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Skip games we've already processed in a previous run
        for _ in range(already_done):
            if chess.pgn.read_game(pgn_f) is None:
                break

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            processed = 0
            start_time = time.time()

            while processed < (GAMES_TO_PROCESS - already_done):
                game = chess.pgn.read_game(pgn_f)
                if game is None:
                    print("Reached end of PGN file.")
                    break

                try:
                    stats = analyze_game(game, engine)
                    row = game_to_row(game, stats)
                    writer.writerow(row)
                    csv_f.flush()  # write to disk immediately, don't lose progress
                except Exception as e:
                    print(f"Skipped a game due to error: {e}")
                    continue

                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = elapsed / processed
                    remaining = (GAMES_TO_PROCESS - already_done - processed) * rate
                    print(f"Processed {already_done + processed}/{GAMES_TO_PROCESS} games "
                          f"| {elapsed/60:.1f} min elapsed "
                          f"| ~{remaining/60:.1f} min remaining")

    print(f"Done. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
