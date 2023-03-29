# Class to decode a one-hot vector into a chess move in the format used by python-chess
# See process_data.py for more information on the data format

import numpy as np
import chess

# from_square_vector, to_square_vector, promotion_vector, capture_vector, check_vector, checkmate_vector
# The following are the possible moves in chess
# the first 8 are the file
# the next 8 are the rank
# the next 4 are the promotion pieces
# the next 2 are the capture and check vectors
# the last 1 is the checkmate vector
# the total move vector length is 23

#class for decoding chess moves
class ChessEncDec:
    def __init__(self):
        self.move_array = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    #decodes a one-hot vector into a chess move in the format used by python-chess
    # returns a move in the format "a2a4", "c4d5"
    def decode_move(self, move_vector):
        from_square_vector = move_vector[:16]
        to_square_vector = move_vector[16:32]

        move = self.decode_square(from_square_vector) + self.decode_square(to_square_vector)

        return move
    
    def decode_square(self, square_vector):
        file = np.argmax(square_vector[:8])
        rank = np.argmax(square_vector[8:16])

        square = self.move_array[file] + str(rank + 1)

        return square
    
    def encode_board(self, fen):
        # get the 2d array of 64 elements
        # each element will be a 1-hot vector of the piece at that position
        # the 65th element will be the value of the position
        # the 66th element will be the value of the best move
        # doesn't need more than five, because the best move is always the first move

        # get chess library board object
        chess_board = chess.Board(fen)

        # get the 2d array of 64 elements
        board = np.zeros((8, 8, 12))

        # get the pieces on the board
        for i in range(8):
            for j in range(8):
                piece = chess_board.piece_at(i * 8 + j)
                if piece is not None:
                    board[i][j][piece.piece_type - 1] = 1
                    if chess_board.turn == chess.WHITE:
                        if piece.color:
                            board[i][j][6] = 1
                        else:
                            board[i][j][7] = 1
                    else:
                        if piece.color:
                            board[i][j][7] = 1
                        else:
                            board[i][j][6] = 1

        return board
    
    def decode_board(self, board):
        # get the 2d array of 64 elements
        # each element will be a 1-hot vector of the piece at that position
        # the 65th element will be the value of the position
        # the 66th element will be the value of the best move
        # doesn't need more than five, because the best move is always the first move

        # get chess library board object
        chess_board = chess.Board('8/8/8/8/8/8/8/8 w - - 0 1')

        # get the 2d array of 64 elements
        for i in range(8):
            for j in range(8):
                piece_vector = board[i][j]
                piece_type = 0
                for k in range(6):
                    if piece_vector[k] == 1:
                        piece_type = k + 1

                piece_color = np.argmax(piece_vector[6:8])
                if piece_type != 0:
                    if piece_color == 0:
                        chess_board.set_piece_at(i * 8 + j, chess.Piece(piece_type, chess.BLACK))
                    else:
                        chess_board.set_piece_at(i * 8 + j, chess.Piece(piece_type, chess.WHITE))

        return chess_board
    
#test code
if __name__ == '__main__':
    encdec = ChessEncDec()
    move_vector = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 1., 0., 1.])
    print(move_vector)
    print(encdec.decode_move(move_vector))
    board_fen = 'rn1qkb1r/1pp2ppp/p3pn2/3p4/2PP2b1/3BPN2/PP3PPP/RNBQK2R w KQkq - 0 6'
    board = encdec.encode_board(board_fen)
    print(board)
    print(encdec.decode_board(board))