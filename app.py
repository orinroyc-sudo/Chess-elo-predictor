"""
Live demo: enter a Lichess username, and this app fetches a few of their
recent games, analyzes them with Stockfish, and predicts their rating
from their playing style.

Run with (NOT python app.py):
    streamlit run app.py
"""

import io
import os
import re
import time

import chess
import chess.engine
import chess.pgn
import joblib
import pandas as pd
import requests
import streamlit as st

_LOCAL_WINDOWS_PATH = (
    r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)
# On Streamlit Cloud (Linux), Stockfish is installed via packages.txt and is
# available as a plain command on PATH. Locally on Windows, we use the
# downloaded exe instead. Check which one actually exists at startup.
STOCKFISH_PATH = _LOCAL_WINDOWS_PATH if os.path.exists(_LOCAL_WINDOWS_PATH) else "stockfish"
MODEL_FILE = "elo_model.joblib"
ANALYSIS_DEPTH = 12
MAX_GAMES = 4

MATE_SCORE = 1000
INACCURACY_THRESHOLD = 50
MISTAKE_THRESHOLD = 100
BLUNDER_THRESHOLD = 300
CLASS_TO_KEY = {"blunder": "blunders", "mistake": "mistakes", "inaccuracy": "inaccuracies"}
OPENING_END_MOVE = 15
MIDDLEGAME_END_MOVE = 40
CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")


# ---------- Feature extraction (same logic as batch_analyze_v2.py) ----------

def score_to_cp(score, pov):
    relative = score.pov(pov)
    if relative.is_mate():
        return MATE_SCORE if relative.mate() > 0 else -MATE_SCORE
    return relative.score()


def classify_move(cpl):
    if cpl >= BLUNDER_THRESHOLD:
        return "blunder"
    if cpl >= MISTAKE_THRESHOLD:
        return "mistake"
    if cpl >= INACCURACY_THRESHOLD:
        return "inaccuracy"
    return None


def phase_for_move_number(n):
    if n <= OPENING_END_MOVE:
        return "opening"
    if n <= MIDDLEGAME_END_MOVE:
        return "middlegame"
    return "endgame"


def parse_time_control(tc):
    if not tc or "+" not in tc:
        return None
    try:
        base, inc = tc.split("+")
        return int(base), int(inc)
    except ValueError:
        return None


def parse_clock_seconds(comment):
    m = CLK_PATTERN.search(comment or "")
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    return h * 3600 + mi * 60 + s


def new_side_stats():
    return {
        "cpl_total": 0, "moves": 0,
        "blunders": 0, "mistakes": 0, "inaccuracies": 0,
        "phase_cpl": {"opening": 0, "middlegame": 0, "endgame": 0},
        "phase_moves": {"opening": 0, "middlegame": 0, "endgame": 0},
        "move_times": [], "prev_clock": None,
    }


def analyze_game(game, engine, increment):
    board = game.board()
    stats = {"white": new_side_stats(), "black": new_side_stats()}
    move_counters = {"white": 0, "black": 0}

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
    return (s["phase_cpl"][phase] / moves) if moves else None


def time_stats(times):
    if not times:
        return None, None
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, var ** 0.5


def single_game_features(game, stats, side_key):
    s = stats[side_key]
    eco = game.headers.get("ECO", "")
    t_mean, t_std = time_stats(s["move_times"])
    return {
        "avg_cpl": s["cpl_total"] / max(1, s["moves"]),
        "blunders": s["blunders"],
        "mistakes": s["mistakes"],
        "inaccuracies": s["inaccuracies"],
        "num_moves": s["moves"],
        "opening_cpl": phase_avg(s, "opening"),
        "middlegame_cpl": phase_avg(s, "middlegame"),
        "endgame_cpl": phase_avg(s, "endgame"),
        "avg_time_per_move": t_mean,
        "time_std": t_std,
        "eco_family": eco[0] if eco else None,
    }


# ---------- Lichess fetching ----------

def fetch_recent_games(username, max_games, max_retries=3):
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "rated": "true",
        "perfType": "blitz,rapid",
        "clocks": "true",
        "opening": "true",
    }
    headers = {
        "User-Agent": "EloPredictorProject/1.0 (student data science project)",
        "Accept": "application/x-chess-pgn",
    }
    token = st.secrets.get("LICHESS_TOKEN")
    if token and token != "paste_your_token_here":
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.text

        if response.status_code == 429:
            wait_seconds = int(response.headers.get("Retry-After", 5))
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
                continue

        raise RuntimeError(
            f"Lichess returned status {response.status_code} for URL:\n"
            f"{response.url}\n\nResponse body:\n{response.text[:500]}"
        )

    raise RuntimeError("Lichess kept rate-limiting the request after several retries.")


def parse_games(pgn_text, username):
    games = []
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()
        if username.lower() == white:
            games.append((game, "white"))
        elif username.lower() == black:
            games.append((game, "black"))
    return games


# ---------- Aggregation + prediction ----------

def aggregate_features(per_game_features, bundle):
    numeric_features = bundle["numeric_features"]
    medians = bundle["medians"]

    row = {}
    for col in numeric_features:
        values = [f[col] for f in per_game_features if f.get(col) is not None]
        row[col] = sum(values) / len(values) if values else medians[col]

    eco_values = [f["eco_family"] for f in per_game_features if f.get("eco_family")]
    most_common_eco = max(set(eco_values), key=eco_values.count) if eco_values else None

    for eco_col in bundle["eco_categories"]:
        row[eco_col] = 1 if eco_col == f"eco_{most_common_eco}" else 0

    return row


def main():
    st.title("Chess Elo Predictor")
    st.write("Enter a Lichess username. We'll analyze their recent blitz/rapid "
             "games with Stockfish and predict their rating from playing style alone.")

    username = st.text_input("Lichess username")
    username = username.strip() if username else username
    run_button = st.button("Analyze")

    if run_button and username:
        try:
            bundle = joblib.load(MODEL_FILE)
        except FileNotFoundError:
            st.error(f"Model file '{MODEL_FILE}' not found. Run train_and_save_model.py first.")
            return

        with st.spinner(f"Fetching games for {username}..."):
            try:
                pgn_text = fetch_recent_games(username, MAX_GAMES)
            except RuntimeError as e:
                st.error(str(e))
                return
            except requests.RequestException as e:
                st.error(f"Couldn't reach Lichess: {e}")
                return

        games = parse_games(pgn_text, username)
        if not games:
            st.error("No recent rated blitz/rapid games found for that username.")
            return

        st.write(f"Found {len(games)} games. Running Stockfish analysis "
                 "(this takes 30-60 seconds)...")

        progress = st.progress(0)
        per_game_features = []
        actual_ratings = []

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            for i, (game, side) in enumerate(games):
                tc = parse_time_control(game.headers.get("TimeControl", ""))
                increment = tc[1] if tc else None

                stats = analyze_game(game, engine, increment)
                per_game_features.append(single_game_features(game, stats, side))

                elo_key = "WhiteElo" if side == "white" else "BlackElo"
                if game.headers.get(elo_key, "").isdigit():
                    actual_ratings.append(int(game.headers[elo_key]))

                progress.progress((i + 1) / len(games))

        feature_row = aggregate_features(per_game_features, bundle)
        X = pd.DataFrame([feature_row])[bundle["feature_columns"]]

        predicted_elo = bundle["model"].predict(X)[0]

        col1, col2 = st.columns(2)
        col1.metric("Predicted Elo", f"{predicted_elo:.0f}")
        if actual_ratings:
            col2.metric("Actual Elo (most recent)", f"{actual_ratings[0]}")

        st.caption("Typical model error is around ±200 rating points -- "
                   "treat this as a rough estimate, not a precise measurement.")

        if predicted_elo >= 2150 or predicted_elo <= 1050:
            st.warning(
                "This prediction is near the edge of the model's training range "
                "(1000-2200). The model was only trained on players in that band, "
                "so predictions for much stronger or weaker players are likely "
                "less reliable -- it may be under- or over-estimating."
            )

        st.subheader("Extracted features (averaged across games analyzed)")
        st.dataframe(pd.DataFrame([feature_row]))


if __name__ == "__main__":
    main()
