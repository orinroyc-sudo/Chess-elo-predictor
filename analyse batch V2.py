"""
Analyze a batch of games from sample_games.pgn with Stockfish, producing
a richer feature set: overall + phase-specific centipawn loss, opening
family, and time-usage stats (if clock data is present).

Saves progress after every game, so it's safe to stop and resume later.

Usage:
    python batch_analyze_v2.py
"""

import csv
import os
import re
import time

import chess
import chess.engine
import chess.pgn

PGN_FILE = "sample_games.pgn"
OUTPUT_CSV = "features_dataset_v2.csv"

STOCKFISH_PATH = (
    r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)

ANALYSIS_DEPTH = 12
GAMES_TO_PROCESS = 1200

MATE_SCORE = 1000

INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD = 100
BLUNDER_THRESHOLD = 300

CLASS_TO_KEY = {
    "blunder": "blunders",
    "mistake": "mistakes",
    "inaccuracy": "inaccuracies",
}

# Game phase boundaries, in half-moves (ply) per side.
# i.e. each side's move number, not total plies in the game.
OPENING_END_MOVE = 15
MIDDLEGAME_END_MOVE = 40

CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")

FIELDNAMES = [
    "white_elo", "black_elo", "eco", "eco_family", "opening", "time_control",
    "result", "num_moves",
    "white_avg_cpl", "black_avg_cpl",
    "white_blunders", "white_mistakes", "white_inaccuracies",
    "black_blunders", "black_mistakes", "black_inaccuracies",
    "white_opening_cpl", "white_middlegame_cpl", "white_endgame_cpl",
    "black_opening_cpl", "black_middlegame_cpl", "black_endgame_cpl",
    "white_avg_time_per_move", "white_time_std",
    "black_avg_time_per_move", "black_time_std",
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


def phase_for_move_number(move_number):
    if move_number <= OPENING_END_MOVE:
        return "opening"
    if move_number <= MIDDLEGAME_END_MOVE:
        return "middlegame"
    return "endgame"


def parse_time_control(tc_header):
    """'180+2' -> (base_seconds=180, increment_seconds=2). Returns None if
    the format isn't recognized (e.g. correspondence games use '-')."""
    if not tc_header or "+" not in tc_header:
        return None
    try:
        base, inc = tc_header.split("+")
        return int(base), int(inc)
    except ValueError:
        return None


def parse_clock_seconds(comment):
    match = CLK_PATTERN.search(comment or "")
    if not match:
        return None
    h, m, s = map(int, match.groups())
    return h * 3600 + m * 60 + s


def new_side_stats():
    return {
        "cpl_total": 0, "moves": 0,
        "blunders": 0, "mistakes": 0, "inaccuracies": 0,
        "phase_cpl": {"opening": 0, "middlegame": 0, "endgame": 0},
        "phase_moves": {"opening": 0, "middlegame": 0, "endgame": 0},
        "move_times": [],
        "prev_clock": None,
    }


def analyze_game(game, engine, increment):
    board = game.board()
    stats = {"white": new_side_stats(), "black": new_side_stats()}

    move_counters = {"white": 0, "black": 0}  # this side's own move number

    for node in game.mainline():
        move = node.move
        mover = board.turn
        side_key = "white" if mover else "black"
        move_counters[side_key] += 1
        move_number = move_counters[side_key]

        info_before = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        eval_before = score_to_cp(info_before["score"], mover)

        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        eval_after = score_to_cp(info_after["score"], mover)

        cpl = max(0, eval_before - eval_after)
        s = stats[side_key]
        s["cpl_total"] += cpl
        s["moves"] += 1

        phase = phase_for_move_number(move_number)
        s["phase_cpl"][phase] += cpl
        s["phase_moves"][phase] += 1

        move_class = classify_move(cpl)
        if move_class:
            s[CLASS_TO_KEY[move_class]] += 1

        # Time usage, if clock data is present in the PGN comment
        if increment is not None:
            clock_now = parse_clock_seconds(node.comment)
            if clock_now is not None and s["prev_clock"] is not None:
                time_spent = s["prev_clock"] - clock_now + increment
                if time_spent >= 0:
                    s["move_times"].append(time_spent)
            s["prev_clock"] = clock_now

    return stats


def phase_avg(s, phase):
    moves = s["phase_moves"][phase]
    return round(s["phase_cpl"][phase] / moves, 1) if moves else ""


def time_stats(times):
    if not times:
        return "", ""
    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    return round(mean, 1), round(variance ** 0.5, 1)


def game_to_row(game, stats):
    headers = game.headers
    w, b = stats["white"], stats["black"]
    eco = headers.get("ECO", "")

    w_time_mean, w_time_std = time_stats(w["move_times"])
    b_time_mean, b_time_std = time_stats(b["move_times"])

    return {
        "white_elo": headers.get("WhiteElo", ""),
        "black_elo": headers.get("BlackElo", ""),
        "eco": eco,
        "eco_family": eco[0] if eco else "",
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
        "white_opening_cpl": phase_avg(w, "opening"),
        "white_middlegame_cpl": phase_avg(w, "middlegame"),
        "white_endgame_cpl": phase_avg(w, "endgame"),
        "black_opening_cpl": phase_avg(b, "opening"),
        "black_middlegame_cpl": phase_avg(b, "middlegame"),
        "black_endgame_cpl": phase_avg(b, "endgame"),
        "white_avg_time_per_move": w_time_mean,
        "white_time_std": w_time_std,
        "black_avg_time_per_move": b_time_mean,
        "black_time_std": b_time_std,
    }


def count_existing_rows():
    if not os.path.exists(OUTPUT_CSV):
        return 0
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    already_done = count_existing_rows()
    if already_done > 0:
        print(f"Resuming: {already_done} games already processed, skipping those.")

    write_header = not os.path.exists(OUTPUT_CSV)
    saw_any_clock_data = False

    with open(PGN_FILE, encoding="utf-8") as pgn_f, \
         open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

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

                tc = parse_time_control(game.headers.get("TimeControl", ""))
                increment = tc[1] if tc else None

                try:
                    stats = analyze_game(game, engine, increment)
                    row = game_to_row(game, stats)
                    if row["white_avg_time_per_move"] != "":
                        saw_any_clock_data = True
                    writer.writerow(row)
                    csv_f.flush()
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

    if not saw_any_clock_data:
        print("\nNote: no clock data (%clk) was found in this PGN sample -- "
              "time-usage columns will be empty. This means your Lichess "
              "download month/format didn't include clock annotations.")

    print(f"Done. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
