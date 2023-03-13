import chess
import chess.engine
from svm_bot.svm_chess_model import next_move, train_model
import pickle





def play_against_stockfish():
    "Given model plays against stockfish, returns True if model wins, False otherwise"
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/15.1/bin/stockfish")
    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(time=0.1))
        if result.move is not None: board.push(result.move)
        else: break
        print(board)
        if board.is_game_over():
            break
        board.push(next_move(board)[0]) # type: ignore

        print(board)

    engine.quit()
    return board.result() == "1-0"

def play_against_stockfish_winrate():
    "Given model repeatedly plays against stockfish, returns winrate of model"
    wins = 0
    for i in range(100):
        if not play_against_stockfish():
            wins += 1
    return wins / 100
    

print(play_against_stockfish_winrate())

import random

class ChessBot:
    def __init__(self, model):
        self.model = model
        self.moves = []

    def move(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        move, legal = next_move(board, self.model)
        if legal:
            board.push(move)
        else:
            move = random.choice(legal_moves)
            board.push(move)

import chess
import time
from IPython.display import clear_output, display
from svm_bot.svm_chess_model import next_move

def play_game(model, should_visualize):
    white = ChessBot(model)
    black = ChessBot(model)
    board = chess.Board()

    if should_visualize:
        display(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            white.move(board)
        else:
            black.move(board)

        if should_visualize:
            clear_output()
            display(board)
            time.sleep(0.5)

play_game(next_move, True)