import pickle
import chess
from .data_handling import position_to_vector
import sklearn
import numpy as np

class SVMBot():
    def __init__(self, model):
        self.model = model
    
    def get_best_move(self, board):
        # if white to move
        best_move_uci = None

        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                prediction = self.model.predict(np.array(position_to_vector(board)).reshape(1, -1))
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                prediction = self.model.predict(np.array(position_to_vector(board)).reshape(1, -1))
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        return best_move_uci
    
