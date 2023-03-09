import chess
import numpy as np

class ModelInput:
    # Organised as [black/white, piece number, existence of piece/rank/file]
    rooks = np.full((2, 2, 3), -1.0)
    knights = np.full((2, 2, 3), -1.0)
    bishops = np.full((2, 2, 3), -1.0)
    queen = np.full((2, 3), -1.0)
    king = np.full((2, 3), -1.0)
    pawns = np.full((2, 8, 3), -1.0)

    def __init__(self, board):
        self.board = board
        self.parse_board()
        
    def parse_board(self):
        rook_index = np.zeros(2, dtype=int)
        knight_index = np.zeros(2, dtype=int)
        bishop_index = np.zeros(2, dtype=int)
        pawn_index = np.zeros(2, dtype=int)

        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            color = int(piece.color)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if piece.piece_type == chess.ROOK:
                self.rooks[color][rook_index[color]] = [1, rank, file]
                rook_index[color] += 1
            elif piece.piece_type == chess.KNIGHT:
                self.knights[color][knight_index[color]] = [1, rank, file]
                knight_index[color] += 1
            elif piece.piece_type == chess.BISHOP:
                self.bishops[color][bishop_index[color]] = [1, rank, file]
                bishop_index[color] += 1
            elif piece.piece_type == chess.QUEEN:
                self.queen[color] = [1, rank, file]
            elif piece.piece_type == chess.KING:
                self.king[color] = [1, rank, file]
            elif piece.piece_type == chess.PAWN:
                self.pawns[color][pawn_index[color]] = [1, rank, file]
                pawn_index[color] += 1

        # sort per piece type according to the rank then the file if rank is the same
        self.rooks = np.sort(self.rooks, axis=1)
        self.knights = np.sort(self.knights, axis=1)
        self.bishops = np.sort(self.bishops, axis=1)
        self.pawns = np.sort(self.pawns, axis=1)

        # Normalise positions between 0 and 1 (0 will represent the position of a piece which has been captured)
        self.rooks[:, :, 1:] = (self.rooks[:, :, 1:] + 1) / 7
        self.knights[:, :, 1:] = (self.knights[:, :, 1:] + 1) / 7
        self.bishops[:, :, 1:] = (self.bishops[:, :, 1:] + 1) / 7
        self.queen[:, 1:] = (self.queen[:, 1:] + 1) / 7
        self.king[:, 1:] = (self.king[:, 1:] + 1) / 7
        self.pawns[:, :, 1:] = (self.pawns[:, :, 1:] + 1) / 7

        # Normalise existence of pieces to be either 0 or 1
        self.rooks[self.rooks == -1] = 0
        self.knights[self.knights == -1] = 0
        self.bishops[self.bishops == -1] = 0
        self.queen[self.queen == -1] = 0
        self.king[self.king == -1] = 0
        self.pawns[self.pawns == -1] = 0

    def get_input(self):
        return np.concatenate([
            self.rooks.flatten(),
            self.knights.flatten(),
            self.bishops.flatten(),
            self.queen.flatten(),
            self.king.flatten(),
            self.pawns.flatten()
        ])

if __name__ == '__main__':
    board = chess.Board()
    model_input = ModelInput(board).get_input()
    print(model_input)
    print(model_input.shape)