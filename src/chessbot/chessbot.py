import random
import numpy as np

class ChessBot:
    def __init__(self, model, exploration_rate=0.0):
        self.model = model
        self.moves_made = []
        self.exploration_rate = exploration_rate
        
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
            model_inputs.append(hash(board.fen())) # TODO: Change to using a better representation
            board.pop()

        return moves, np.array(model_inputs).reshape(-1, 1)
    
    def get_best_move(self, moves, model_inputs):
        predictions = self.model.predict(model_inputs)
        move_index = predictions.argmax()
        return model_inputs[move_index], moves[move_index]