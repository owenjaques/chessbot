import chess
import numpy as np

class ModelInput:
    # Organised as [black/white, piece number, rank/file]
    rooks = np.full((2, 2, 2), -1)
    knights = np.full((2, 2, 2), -1)
    bishops = np.full((2, 2, 2), -1)
    queen = np.full((2, 2), -1)
    king = np.full((2, 2), -1)
    pawns = np.full((2, 8, 2), -1)

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
                self.rooks[color][rook_index[color]] = [rank, file]
                rook_index[color] += 1
            elif piece.piece_type == chess.KNIGHT:
                self.knights[color][knight_index[color]] = [rank, file]
                knight_index[color] += 1
            elif piece.piece_type == chess.BISHOP:
                self.bishops[color][bishop_index[color]] = [rank, file]
                bishop_index[color] += 1
            elif piece.piece_type == chess.QUEEN:
                self.queen[color] = [rank, file]
            elif piece.piece_type == chess.KING:
                self.king[color] = [rank, file]
            elif piece.piece_type == chess.PAWN:
                self.pawns[color][pawn_index[color]] = [rank, file]
                pawn_index[color] += 1

    def get_input(self):
        return np.array([
            self.rooks,
            self.knights,
            self.bishops,
            self.queen,
            self.king,
            self.pawns
        ])

if __name__ == '__main__':
    board = chess.Board()
    model_input = ModelInput(board)
    print(model_input)