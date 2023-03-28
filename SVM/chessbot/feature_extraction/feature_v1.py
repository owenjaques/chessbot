"""
This approach is known as board representation, 
and it involves representing each square on the board as a feature, 
with a value indicating the type of piece on that square (e.g., pawn, knight, bishop, rook, queen, king) 
and which color the piece belongs to.
"""
import chess
import numpy as np

# given a board, return a 1D array of 64 features
# piece value: > 0 if white, < 0 if black (0 if empty)
# piece type: 1 for pawn, 2 for knight, 3 for bishop, 4 for rook, 5 for queen, 6 for king


def get_features(board):
    """
    Returns a 1D array of 64 features.
    """
    features = []
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            features.append(0)
        else:
            features.append(piece.piece_type * (1 if piece.color else -1))

    return np.array(features)


# test_board = chess.Board()
# print(get_features(test_board))