import random
import numpy as np
import chess
from .model_input import ModelInput

class ChessBot:
    def __init__(self, model, color, exploration_rate=0.0):
        self.model = model
        self.moves_made = []
        self.exploration_rate = exploration_rate
        self.color = color
        
    def move(self, board):
        moves, model_inputs = self.get_potential_moves(board)
        
        has_fit = hasattr(self.model, 'n_iter_')
        if not has_fit or random.random() < self.exploration_rate:
            move_index, move = random.choice(list(enumerate(moves)))
            model_input = model_inputs[move_index]
        else:
            model_input, move = self.get_best_move(moves, model_inputs)
        
        self.moves_made.append(model_input)
        
        return move

    def get_potential_moves(self, board):
        moves = list(board.legal_moves)
        model_inputs = []

        for move in moves:
            board.push(move)
            model_inputs.append(ModelInput(board).get_input())
            board.pop()

        return moves, np.array(model_inputs)
    
    def get_best_move(self, moves, model_inputs):
        predictions = self.model.predict(model_inputs)
        move_index = predictions.argmax() if self.color == chess.WHITE else predictions.argmin()
        return model_inputs[move_index], moves[move_index]