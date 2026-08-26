"""
Stream a Lichess monthly game dump and stop once we've collected
enough games, without downloading the whole (multi-GB) file.

Usage:
    python collect_games.py
"""

import io
import chess.pgn
import requests
import zstandard as zstd

# Pick any month from https://database.lichess.org
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst"

TARGET_GAME_COUNT = 5000       # stop once we have this many qualifying games
MIN_RATING, MAX_RATING = 1000, 2200
ALLOWED_TIME_CONTROLS = {"Blitz", "Rapid"}
MIN_MOVES = 15

OUTPUT_FILE = "sample_games.pgn"


def qualifies(game) -> bool:
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return False

    if not (MIN_RATING <= white_elo <= MAX_RATING):
        return False
    if not (MIN_RATING <= black_elo <= MAX_RATING):
        return False

    time_control = headers.get("Event", "")
    if not any(tc in time_control for tc in ALLOWED_TIME_CONTROLS):
        return False

    # rough proxy for game length before fully parsing moves
    if len(list(game.mainline_moves())) < MIN_MOVES:
        return False

    return True


def stream_games():
    collected = 0
    scanned = 0

    with requests.get(URL, stream=True) as response:
        response.raise_for_status()

        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
                while collected < TARGET_GAME_COUNT:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break  # ran out of games (shouldn't happen this early)

                    scanned += 1
                    if qualifies(game):
                        out.write(str(game) + "\n\n")
                        collected += 1

                    if scanned % 2000 == 0:
                        print(f"scanned {scanned:,} games, kept {collected:,}")

    print(f"Done. Scanned {scanned:,} games, saved {collected:,} to {OUTPUT_FILE}")
    print("Connection closed early — the rest of the multi-GB file was never downloaded.")


if __name__ == "__main__":
    stream_games()
