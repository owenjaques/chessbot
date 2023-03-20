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

    #Attacked squares 
    attacks = np.zeros(64)

    # Miscenllaneous data that is useful for the model
    castling_rights = np.empty(4)
    potential_game_outcome = np.empty(1)
    next_to_move = np.empty(1)

    def __init__(self, board):
        self.board = board
        self.parse_board()
        self.parse_castling_rights()
        self.parse_potential_game_outcome()
        self.get_next_to_move()
        self.atk_lst()
        
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

            if piece.piece_type == chess.ROOK and rook_index[color] < 2:
                self.rooks[color][rook_index[color]] = [1, rank, file]
                rook_index[color] += 1
            elif piece.piece_type == chess.KNIGHT and knight_index[color] < 2:
                self.knights[color][knight_index[color]] = [1, rank, file]
                knight_index[color] += 1
            elif piece.piece_type == chess.BISHOP and bishop_index[color] < 2:
                self.bishops[color][bishop_index[color]] = [1, rank, file]
                bishop_index[color] += 1
            elif piece.piece_type == chess.QUEEN and self.queen[color][0] == -1:
                self.queen[color] = [1, rank, file]
            elif piece.piece_type == chess.KING:
                self.king[color] = [1, rank, file]
            elif piece.piece_type == chess.PAWN:
                self.pawns[color][pawn_index[color]] = [1, rank, file]
                pawn_index[color] += 1


        # Normalise positions between 0 and 1 (0 will represent the position of a piece which has been captured)
        self.rooks[:, :, 1:] = (self.rooks[:, :, 1:] + 1) / 8
        self.knights[:, :, 1:] = (self.knights[:, :, 1:] + 1) / 8
        self.bishops[:, :, 1:] = (self.bishops[:, :, 1:] + 1) / 8
        self.queen[:, 1:] = (self.queen[:, 1:] + 1) / 8
        self.king[:, 1:] = (self.king[:, 1:] + 1) / 8
        self.pawns[:, :, 1:] = (self.pawns[:, :, 1:] + 1) / 8

        # Normalise existence of pieces to be either 0 or 1
        self.rooks[self.rooks == -1] = 0
        self.knights[self.knights == -1] = 0
        self.bishops[self.bishops == -1] = 0
        self.queen[self.queen == -1] = 0
        self.king[self.king == -1] = 0
        self.pawns[self.pawns == -1] = 0

    def parse_castling_rights(self):
        self.castling_rights[0] = int(self.board.has_kingside_castling_rights(chess.WHITE))
        self.castling_rights[1] = int(self.board.has_queenside_castling_rights(chess.WHITE))
        self.castling_rights[2] = int(self.board.has_kingside_castling_rights(chess.BLACK))
        self.castling_rights[3] = int(self.board.has_queenside_castling_rights(chess.BLACK))

    def parse_potential_game_outcome(self):
        game_outcome = self.board.outcome(claim_draw=True)
        if game_outcome is None:
            self.potential_game_outcome[0] = 0.66
        else:
            if (game_outcome.winner == chess.WHITE and self.board.ply() % 2 == 1) or (game_outcome.winner == chess.BLACK and self.board.ply() % 2 == 0):
                # win
                self.potential_game_outcome[0] = 1
            elif (game_outcome.winner == chess.WHITE and self.board.ply() % 2 == 0) or (game_outcome.winner == chess.BLACK and self.board.ply() % 2 == 1):
                # loss
                self.potential_game_outcome[0] = 0
            else:
                # draw
                self.potential_game_outcome[0] = 0.33

    def get_next_to_move(self):
        self.next_to_move[0] = 0 if self.board.turn else 1


    def atk_lst(self):
        white_atk = chess.SquareSet()
        black_atk = chess.SquareSet()
        for attacker in chess.SquareSet(self.board.occupied_co[chess.WHITE]):
            white_atk |= self.board.attacks(attacker)
        for attacker in chess.SquareSet(self.board.occupied_co[chess.BLACK]):
            black_atk |= self.board.attacks(attacker)

        for i in list(white_atk):
            self.attacks[i] += (1/3)
        
        for i in list(black_atk):
            self.attacks[i] += (2/3)
        


    def get_input(self):
        return np.concatenate([
            self.rooks.flatten(),
            self.knights.flatten(),
            self.bishops.flatten(),
            self.queen.flatten(),
            self.king.flatten(),
            self.pawns.flatten(),
            self.castling_rights,
            self.potential_game_outcome,
            self.attacks,
            self.next_to_move
        ])

if __name__ == '__main__':
    board = chess.Board()
    board.push(chess.Move.from_uci('e2e4'))
    model_input = ModelInput(board).get_input()
    print(model_input)
    print(model_input.shape)