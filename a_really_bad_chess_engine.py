#This file is a placeholder for our engines.  It was created to test the elo calculation using a known stockfish elo.
import random
from stockfish import Stockfish
import chess

def get_next_move(board):
    #Changed the engine to be stonger for more thorough testing of the elo estimation
    stockfish = Stockfish(path = "/usr/local/bin/stockfish", parameters = {"UCI_LimitStrength" : "true", "UCI_Elo" : 2000})
    stockfish.set_fen_position(board.fen())
    return chess.Move.from_uci(stockfish.get_best_move())