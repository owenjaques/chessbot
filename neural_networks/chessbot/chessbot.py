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
        model_input = ModelInput(board).get_input()
        board.pop()
        return model_input
    
    def get_best_move(self, moves, board):
        model_inputs = []

        for move in moves:
            model_inputs.append(self.convert_move_to_model_input(board, move))

        predictions = self.model.predict(np.array(model_inputs))       
        color_idx = 2 if self.color == chess.WHITE else 0
        move_index = predictions[:,color_idx].argmax()

        return model_inputs[move_index], moves[move_index]