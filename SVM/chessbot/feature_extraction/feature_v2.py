"""
This approach use game-specific features, 
such as the number of pieces captured by each player, 
the number of legal moves available to each player, 
and the presence or absence of specific pawn structures or opening moves.
"""



import chess
import numpy as np
import random

def num_piece_capture(board: chess.Board, piece_type: int, color: bool) -> int:
    """Returns the number of pieces of the given type and color captured by the given color.

    Args:
        board: The board to check.
        piece_type: The type of piece to check for.
        color: The color of the pieces to check for.

    Returns:
        The number of pieces of the given type and color captured by the given color.
    """
    return len(board.pieces(piece_type, not color)) - len(board.pieces(piece_type, color))

def num_legal_moves(board: chess.Board, color: bool) -> int:
    """Returns the number of legal moves available to the given color.

    Args:
        board: The board to check.
        color: The color to check for.

    Returns:
        The number of legal moves available to the given color.
    """
    return len(list(board.legal_moves))

def get_features(board: chess.Board) -> np.ndarray:
    """Returns a 1D array of 14 features.
    """
    features = []
    for piece_type in range(1, 7):
        features.append(num_piece_capture(board, piece_type, True))
        features.append(num_piece_capture(board, piece_type, False))
    features.append(num_legal_moves(board, True))
    features.append(num_legal_moves(board, False))
    return np.array(features)

'''test_board = chess.Board()
# generate random board state
for _ in range(20):
    random_move = random.choice(list(test_board.legal_moves))
    test_board.push(random_move) 
print(get_features(test_board))'''

