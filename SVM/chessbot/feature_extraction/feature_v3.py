import chess
import numpy as np

def get_board_position(board):
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

    return features

def get_turn(board):
    """
    Returns a 1D array of 1 feature.
    """
    return [1 if board.turn else -1] # 1 if white's turn, -1 if black's turn

def get_material_count(board):
    """
    Returns a 1D array of count of each piece type.
    """
    features = [0] * 6
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            continue
        features[piece.piece_type - 1] += 1 * (1 if piece.color else -1)

    return features # [white pawn, white knight, white bishop, white rook, white queen, white king, black pawn, black knight, black bishop, black rook, black queen, black king]

def get_material_value(board):
    """
    Returns a 1D array of value of each piece type.
    """
    features = [0] * 6
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            continue
        features[piece.piece_type - 1] += piece.piece_type * (1 if piece.color else -1)

    return features # [white pawn, white knight, white bishop, white rook, white queen, white king, black pawn, black knight, black bishop, black rook, black queen, black king]

def get_features(board):
    """
    Returns a 1D array of 73 features.
    """
    features = get_board_position(board)
    features.extend(get_turn(board))
    features.extend(get_material_count(board))
    features.extend(get_material_value(board))
    return np.array(features)


if __name__ == "__main__":
    # test
    test_board = chess.Board()
    print(get_features(test_board))
