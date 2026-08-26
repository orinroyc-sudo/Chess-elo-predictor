"""
Analyze ONE game from sample_games.pgn with Stockfish, and print the
centipawn loss for every move. This is a test run before scaling up
to all 5,000 games.

Usage:
    python analyze_single_game.py
"""

import time

import chess
import chess.engine
import chess.pgn

PGN_FILE = "sample_games.pgn"

# Update this path if your folder names differ
STOCKFISH_PATH = (
    r"stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)

# How hard Stockfish thinks per move. Higher = more accurate but slower.
ANALYSIS_DEPTH = 12

MATE_SCORE = 1000  # how a forced mate gets converted into a centipawn-ish number


def score_to_cp(score, pov):
    """Convert a python-chess score object into a centipawn number
    from the given side's point of view."""
    relative = score.pov(pov)
    if relative.is_mate():
        # being mated is very bad, delivering mate is very good
        mate_in = relative.mate()
        return MATE_SCORE if mate_in > 0 else -MATE_SCORE
    return relative.score()


def analyze_game(game, engine):
    board = game.board()
    move_data = []

    for move_number, move in enumerate(game.mainline_moves(), start=1):
        mover = board.turn  # True = White, False = Black

        # Evaluate the position BEFORE the move (best case for the mover)
        info_before = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        best_move = info_before["pv"][0] if "pv" in info_before else None
        eval_before = score_to_cp(info_before["score"], mover)

        # Play the actual move, then evaluate again
        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        # score.pov(mover) already converts to the mover's perspective correctly,
        # regardless of whose turn it is now — no extra sign flip needed here.
        eval_after = score_to_cp(info_after["score"], mover)

        centipawn_loss = max(0, eval_before - eval_after)

        move_data.append({
            "move_number": move_number,
            "mover": "White" if mover else "Black",
            "move_played": move.uci(),
            "best_move": best_move.uci() if best_move else None,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "centipawn_loss": centipawn_loss,
        })

    return move_data


def main():
    with open(PGN_FILE, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)  # just the first game in the file

    print(f"Analyzing: {game.headers.get('White')} vs {game.headers.get('Black')}")
    print(f"White Elo: {game.headers.get('WhiteElo')}  Black Elo: {game.headers.get('BlackElo')}")
    print(f"Depth: {ANALYSIS_DEPTH}\n")

    start_time = time.time()
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        move_data = analyze_game(game, engine)
    elapsed = time.time() - start_time

    total_white_cpl = sum(m["centipawn_loss"] for m in move_data if m["mover"] == "White")
    total_black_cpl = sum(m["centipawn_loss"] for m in move_data if m["mover"] == "Black")
    white_moves = sum(1 for m in move_data if m["mover"] == "White")
    black_moves = sum(1 for m in move_data if m["mover"] == "Black")

    for m in move_data[:10]:  # just show the first 10 moves so it's not a wall of text
        print(f"{m['move_number']:>3} {m['mover']:<5} played {m['move_played']:<7} "
              f"(best: {m['best_move']:<7}) cp loss: {m['centipawn_loss']}")

    print("\n--- Summary ---")
    print(f"White avg centipawn loss: {total_white_cpl / white_moves:.1f}")
    print(f"Black avg centipawn loss: {total_black_cpl / black_moves:.1f}")

    num_moves_total = len(move_data)
    print("\n--- Timing ---")
    print(f"This game: {elapsed:.1f} seconds for {num_moves_total} moves "
          f"({elapsed / num_moves_total:.2f} sec/move)")
    estimated_5000_hours = (elapsed / num_moves_total) * 5000 * 40 / 3600
    print(f"Rough estimate for 5,000 games (assuming ~40 moves/game avg): "
          f"{estimated_5000_hours:.1f} hours")


if __name__ == "__main__":
    main()
