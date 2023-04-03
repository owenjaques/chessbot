import random
import numpy as np
import chess
from .modelinput import ModelInput

class ChessBot:
    def __init__(self, model, model_input, color, exploration_rate=0.0):
        self.model = model
        self.model_input = model_input
        self.moves_made = []
        self.exploration_rate = exploration_rate
        self.color = color
        
    def move(self, board):
        moves = list(board.legal_moves)
        
        if random.random() < self.exploration_rate:
            move = random.choice(moves)
            model_input = self.convert_move_to_model_input(board, move)
        else:
            model_input, move = self.get_best_move(moves, board)
        
        self.moves_made.append(model_input)
        
        return move

    def convert_move_to_model_input(self, board, move):
        board.push(move)
        model_input = self.model_input.get_input(board)
        board.pop()
        return model_input
    
    def get_best_move(self, moves, board):
        model_inputs = []

        for move in moves:
            model_inputs.append(self.convert_move_to_model_input(board, move))

        predictions = self.model.predict(np.array(model_inputs))       
        move_index = np.argmax(predictions) if self.color == chess.WHITE else np.argmin(predictions)

        return model_inputs[move_index], moves[move_index]