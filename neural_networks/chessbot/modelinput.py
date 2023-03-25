import chess
import numpy as np

def piece_to_index(piece):
    if piece.piece_type == chess.PAWN:
        idx = 0
    elif piece.piece_type == chess.KNIGHT:
        idx = 1
    elif piece.piece_type == chess.BISHOP:
        idx = 2
    elif piece.piece_type == chess.ROOK:
        idx = 3
    elif piece.piece_type == chess.QUEEN:
        idx = 4
    elif piece.piece_type == chess.KING:
        idx = 5
    if piece.color == chess.BLACK:
        idx += 6
    return idx

class ModelInput:
    # Organised as [black/white, piece number, existence of piece/rank/file]
    rooks = np.empty((2, 2, 3))
    knights = np.empty((2, 2, 3))
    bishops = np.empty((2, 2, 3))
    queen = np.empty((2, 3))
    king = np.empty((2, 3))
    pawns = np.empty((2, 8, 3))

    #Attacked squares 
    attacks = np.empty(64)

    # Miscenllaneous data that is useful for the model
    castling_rights = np.empty(4)
    potential_game_outcome = np.empty(1)
    next_to_move = np.empty(1)

    def __init__(self, input_type='positions'):
        self.input_type = input_type

    def parse_simple_board(self, board):
        simple_board = np.zeros((8, 8, 12))

        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            simple_board[rank, file, piece_to_index(piece)] = 1

        return simple_board
        
    def parse_board(self, board):
        self.rooks.fill(0)
        self.knights.fill(0)
        self.bishops.fill(0)
        self.queen.fill(0)
        self.king.fill(0)
        self.pawns.fill(0)

        rook_index = np.zeros(2, dtype=int)
        knight_index = np.zeros(2, dtype=int)
        bishop_index = np.zeros(2, dtype=int)
        pawn_index = np.zeros(2, dtype=int)

        piece_map = board.piece_map()
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

        # Normalise positions between 0 and 1
        self.rooks[:, :, 1:] = (self.rooks[:, :, 1:]) / 7
        self.knights[:, :, 1:] = (self.knights[:, :, 1:]) / 7
        self.bishops[:, :, 1:] = (self.bishops[:, :, 1:]) / 7
        self.queen[:, 1:] = (self.queen[:, 1:]) / 7
        self.king[:, 1:] = (self.king[:, 1:]) / 7
        self.pawns[:, :, 1:] = (self.pawns[:, :, 1:]) / 7

        # Set positions of pieces that don't exist to -1
        self.rooks[self.rooks[:, :, 0] == 0, 1:] = -1
        self.knights[self.knights[:, :, 0] == 0, 1:] = -1
        self.bishops[self.bishops[:, :, 0] == 0, 1:] = -1
        self.queen[self.queen[:, 0] == 0, 1:] = -1
        self.king[self.king[:, 0] == 0, 1:] = -1
        self.pawns[self.pawns[:, :, 0] == 0, 1:] = -1

    def parse_castling_rights(self, board):
        self.castling_rights[0] = int(board.has_kingside_castling_rights(chess.WHITE))
        self.castling_rights[1] = int(board.has_queenside_castling_rights(chess.WHITE))
        self.castling_rights[2] = int(board.has_kingside_castling_rights(chess.BLACK))
        self.castling_rights[3] = int(board.has_queenside_castling_rights(chess.BLACK))

    def parse_potential_game_outcome(self, board):
        game_outcome = board.outcome(claim_draw=True)
        if game_outcome is None:
            self.potential_game_outcome[0] = 0.66
        else:
            if (game_outcome.winner == chess.WHITE and board.ply() % 2 == 1) or (game_outcome.winner == chess.BLACK and board.ply() % 2 == 0):
                # win
                self.potential_game_outcome[0] = 1
            elif (game_outcome.winner == chess.WHITE and board.ply() % 2 == 0) or (game_outcome.winner == chess.BLACK and board.ply() % 2 == 1):
                # loss
                self.potential_game_outcome[0] = 0
            else:
                # draw
                self.potential_game_outcome[0] = 0.33

    def get_next_to_move(self, board):
        self.next_to_move[0] = 0 if board.turn else 1

    def atk_lst(self, board):
        self.attacks.fill(0)
        white_atk = chess.SquareSet()
        black_atk = chess.SquareSet()
        for attacker in chess.SquareSet(board.occupied_co[chess.WHITE]):
            white_atk |= board.attacks(attacker)
        for attacker in chess.SquareSet(board.occupied_co[chess.BLACK]):
            black_atk |= board.attacks(attacker)

        for i in list(white_atk):
            self.attacks[i] += (1/3)
        
        for i in list(black_atk):
            self.attacks[i] += (2/3)

    def get_flattened_positions(self):
        return np.concatenate([
                self.rooks.flatten(),
                self.knights.flatten(),
                self.bishops.flatten(),
                self.queen.flatten(),
                self.king.flatten(),
                self.pawns.flatten()
            ])

    def get_misc_features(self):
        return np.concatenate([
                self.castling_rights,
                self.potential_game_outcome,
                self.next_to_move
            ])

    def parse_misc_features(self, board):
        self.parse_castling_rights(board)
        self.parse_potential_game_outcome(board)
        self.get_next_to_move(board)

    def get_input_from_fen(self, fen):
        return self.get_input(chess.Board(fen))
        
    def get_input(self, board):
        if self.input_type == 'simple':
            return self.parse_simple_board(board).flatten()
        
        self.parse_board(board)
        if self.input_type == 'positions':
            return self.get_flattened_positions()
        
        self.parse_misc_features(board)
        self.atk_lst(board)
        return self.get_flattened_positions(), self.attacks, self.get_misc_features()

    def input_length(self):
        if self.input_type == 'simple':
            return self.parse_simple_board(chess.Board()).flatten().shape[0]
        
        if self.input_type == 'positions':
            return self.get_flattened_positions().shape[0]
        
        return np.array([self.get_flattened_positions().shape[0], self.attacks.shape[0], self.get_misc_features().shape[0]])

if __name__ == '__main__':
    board = chess.Board()
    board.push(chess.Move.from_uci('e2e4'))
    model_input = ModelInput('simple').get_input(board)
    print(ModelInput('simple').input_length())
    print(model_input)