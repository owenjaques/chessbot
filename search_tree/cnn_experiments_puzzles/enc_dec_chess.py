import chess
import chess.pgn
import numpy as np
import pandas as pd


def encode_move(move):
    # encode chess move as a 1-hot vector of length 64*64
    # move is a string in the form 'e2e4'
    # returns a 1-hot vector of length 64*64

    # convert move to a tuple of ints
    move = (int(move[1])-1)*8 + ord(move[0])-97, (int(move[3])-1)*8 + ord(move[2])-97

    # encode move as a 1-hot vector
    move = np.eye(64*64, dtype=np.int8)[move[0]*64 + move[1]]

    return move

def encode_move_number(move):
    # encode move as a single integer
    # move is a string in the form 'e2e4'
    # returns an integer

    # convert move to a tuple of ints
    move = (int(move[1])-1)*8 + ord(move[0])-97, (int(move[3])-1)*8 + ord(move[2])-97

    # encode move as a single integer
    move = move[0]*64 + move[1]

    return move

def encode_board(board, current_color):
    # encode chess board as a 1-hot vector of length 64*2
    # board is a chess.Board object
    # returns a 1-hot vector of length 64*2
    # needs to place 1s in the squares for current move's color
    # if current_color is True, then white is the current move's color
    # if current_color is False, then black is the current move's color
    # white goes in the first 64 spots, black goes in the second 64 spots

    # initialize board_image
    board_image = np.zeros(64*2, dtype=np.int8)

    white_board = np.zeros(64, dtype=np.int8)
    black_board = np.zeros(64, dtype=np.int8)

    # for square in board
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            white_board[i] = 0
            black_board[i] = 0
        else:
            if piece.color == True:
                white_board[i] = 1
            else:
                black_board[i] = 1

    # if current_color is False, flip the boards
    if current_color == False:
        white_board, black_board = black_board, white_board
        white_board = np.flip(white_board)
        black_board = np.flip(black_board)

    # concatenate the boards
    board_image = np.concatenate((white_board, black_board))

    return board_image

# encode board as a 1-hot vector of length 64*6
def encode_board_six(board, current_color):
    # initialize board_images
    board_image = np.zeros(64*6, dtype=np.int8)

    pawn_board = np.zeros(64, dtype=np.int8)
    knight_board = np.zeros(64, dtype=np.int8)
    bishop_board = np.zeros(64, dtype=np.int8)
    rook_board = np.zeros(64, dtype=np.int8)
    queen_board = np.zeros(64, dtype=np.int8)
    king_board = np.zeros(64, dtype=np.int8)

    # for square in board
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is not None:
            turn = -1
            if piece.color == True:
                turn = 1
            if piece.piece_type == 1:
                pawn_board[i] = 1*turn
            elif piece.piece_type == 2:
                knight_board[i] = 1*turn
            elif piece.piece_type == 3:
                bishop_board[i] = 1*turn
            elif piece.piece_type == 4:
                rook_board[i] = 1*turn
            elif piece.piece_type == 5:
                queen_board[i] = 1*turn
            elif piece.piece_type == 6:
                king_board[i] = 1*turn

    # if current_color is False, flip the boards
    if current_color == False:
        pawn_board = np.flip(pawn_board)
        knight_board = np.flip(knight_board)
        bishop_board = np.flip(bishop_board)
        rook_board = np.flip(rook_board)
        queen_board = np.flip(queen_board)
        king_board = np.flip(king_board)

    # concatenate the boards
    board_image = np.concatenate(( pawn_board, knight_board, bishop_board, rook_board, queen_board, king_board))

    return board_image


# encode board as a 1-hot vector of length 64*8
def encode_board_eight(board, current_color):
    # initialize board_images
    board_image = np.zeros(64*8, dtype=np.int8)

    white_board = np.zeros(64, dtype=np.int8)
    black_board = np.zeros(64, dtype=np.int8)
    pawn_board = np.zeros(64, dtype=np.int8)
    knight_board = np.zeros(64, dtype=np.int8)
    bishop_board = np.zeros(64, dtype=np.int8)
    rook_board = np.zeros(64, dtype=np.int8)
    queen_board = np.zeros(64, dtype=np.int8)
    king_board = np.zeros(64, dtype=np.int8)

    # for square in board
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            white_board[i] = 0
            black_board[i] = 0
        else:
            if piece.color == True:
                white_board[i] = 1
            else:
                black_board[i] = 1
            if piece.piece_type == 1:
                pawn_board[i] = 1
            elif piece.piece_type == 2:
                knight_board[i] = 1
            elif piece.piece_type == 3:
                bishop_board[i] = 1
            elif piece.piece_type == 4:
                rook_board[i] = 1
            elif piece.piece_type == 5:
                queen_board[i] = 1
            elif piece.piece_type == 6:
                king_board[i] = 1

    # if current_color is False, flip the boards
    if current_color == False:
        white_board, black_board = black_board, white_board
        white_board = np.flip(white_board)
        black_board = np.flip(black_board)
        pawn_board = np.flip(pawn_board)
        knight_board = np.flip(knight_board)
        bishop_board = np.flip(bishop_board)
        rook_board = np.flip(rook_board)
        queen_board = np.flip(queen_board)
        king_board = np.flip(king_board)

    # concatenate the boards
    board_image = np.concatenate((white_board, black_board, pawn_board, knight_board, bishop_board, rook_board, queen_board, king_board))

    return board_image

# encode board as a 1-hot vector of length 64*8
def encode_board_twelve(board, current_color):
    # initialize board_images
    board_image = np.zeros(64*12, dtype=np.int8)

    w_pawn_board = np.zeros(64, dtype=np.int8)
    w_knight_board = np.zeros(64, dtype=np.int8)
    w_bishop_board = np.zeros(64, dtype=np.int8)
    w_rook_board = np.zeros(64, dtype=np.int8)
    w_queen_board = np.zeros(64, dtype=np.int8)
    w_king_board = np.zeros(64, dtype=np.int8)

    b_pawn_board = np.zeros(64, dtype=np.int8)
    b_knight_board = np.zeros(64, dtype=np.int8)
    b_bishop_board = np.zeros(64, dtype=np.int8)
    b_rook_board = np.zeros(64, dtype=np.int8)
    b_queen_board = np.zeros(64, dtype=np.int8)
    b_king_board = np.zeros(64, dtype=np.int8)

    # for square in board
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == True:
                if piece.piece_type == 1:
                    w_pawn_board[i] = 1
                elif piece.piece_type == 2:
                    w_knight_board[i] = 1
                elif piece.piece_type == 3:
                    w_bishop_board[i] = 1
                elif piece.piece_type == 4:
                    w_rook_board[i] = 1
                elif piece.piece_type == 5:
                    w_queen_board[i] = 1
                elif piece.piece_type == 6:
                    w_king_board[i] = 1
            else:
                if piece.piece_type == 1:
                    b_pawn_board[i] = 1
                elif piece.piece_type == 2:
                    b_knight_board[i] = 1
                elif piece.piece_type == 3:
                    b_bishop_board[i] = 1
                elif piece.piece_type == 4:
                    b_rook_board[i] = 1
                elif piece.piece_type == 5:
                    b_queen_board[i] = 1
                elif piece.piece_type == 6:
                    b_king_board[i] = 1

    # if current_color is False, flip the boards
    if current_color == False:
        w_pawn_board, b_pawn_board = b_pawn_board, w_pawn_board
        w_knight_board, b_knight_board = b_knight_board, w_knight_board
        w_bishop_board, b_bishop_board = b_bishop_board, w_bishop_board
        w_rook_board, b_rook_board = b_rook_board, w_rook_board
        w_queen_board, b_queen_board = b_queen_board, w_queen_board
        w_king_board, b_king_board = b_king_board, w_king_board

        w_pawn_board = np.flip(w_pawn_board)
        b_pawn_board = np.flip(b_pawn_board)
        w_knight_board = np.flip(w_knight_board)
        b_knight_board = np.flip(b_knight_board)
        w_bishop_board = np.flip(w_bishop_board)
        b_bishop_board = np.flip(b_bishop_board)
        w_rook_board = np.flip(w_rook_board)
        b_rook_board = np.flip(b_rook_board)
        w_queen_board = np.flip(w_queen_board)
        b_queen_board = np.flip(b_queen_board)
        w_king_board = np.flip(w_king_board)
        b_king_board = np.flip(b_king_board)

    # concatenate the boards
    board_image = np.concatenate((w_pawn_board, w_knight_board, w_bishop_board, w_rook_board, w_queen_board, w_king_board, b_pawn_board, b_knight_board, b_bishop_board, b_rook_board, b_queen_board, b_king_board))

    return board_image

# encode game from fen string and move string
def encode_game_fen(fen, move, dim=2):
    # encode chess game as a 1-hot vector of length 64*2 + 64*64
    # fen is a string in the form 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    # move is a string in the form 'e2e4'
    # returns a 1-hot vector of length 64*2 + 64*64

    # initialize board_image
    board_image = np.zeros(64*dim + 64*64)

    # get board from fen
    board = chess.Board(fen)

    # encode board
    if dim == 2:
        board_image[:128] = encode_board(board, board.turn)
    elif dim == 6:
        board_image[:384] = encode_board_six(board, board.turn)
    elif dim == 8:
        board_image[:512] = encode_board_eight(board, board.turn)
    elif dim == 12:
        board_image[:768] = encode_board_twelve(board, board.turn)
    

    # encode move
    board_image[64*dim:] = encode_move(move)

    return board_image


def decode_move(move):
    # decode a 1-hot vector of length 64*64 to a chess move
    # move is a numpy array of length 64*64
    # returns a string in the form 'e2e4'

    # get the index of the 1 in the move
    move = np.where(move == 1)[0][0]

    # decode move
    move = chr(move%64 + 97) + str(move//64 + 1)

    return move



def main():
    # initialize board
    board = chess.Board()

if __name__ == '__main__':
    main()

