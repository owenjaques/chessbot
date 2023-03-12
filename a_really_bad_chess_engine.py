import random

def get_next_move(board):
    return random.choice([move for move in board.legal_moves])