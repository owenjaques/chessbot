# class to process the data the chess puzzle data set into a format that can be used by the neural network
# The data will be processed into various formats that can be tested to see which one works best
# Once processed the data will be saved to a file so that it can be loaded quickly for training

# Should use stockfish to get the best moves for each position
# Should use stockfish to get the value of each position

# The data will be processed into a 2d array of 64 elements


from stockfish import Stockfish
import numpy as np
import pandas as pd
import os
import pickle
import time
import chess

class ProcessData:
    def __init__(self, save_dir, save_file_name, save_file_extension):
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.save_file_extension = save_file_extension
        self.save_file = os.path.join(self.save_dir, self.save_file_name + self.save_file_extension)

    def load_data(self, data_dir, data_file,):
        # load the data from the csv file
        data = pd.read_csv(os.path.join(data_dir, data_file), header=None)
        return data

    def process_data(self, data):
        # process the data into a format that can be used by the neural network
        # the data will be processed into a 2d array of 64 elements
        # each element will be a 1-hot vector of the piece at that position
        # the 65th element will be the value of the position
        # the 66th element will be the value of the best move
        # doesn't need more than five, because the best move is always the first move

        # the data will be processed into a 2d array of 64 elements
        # each element will be a 1-hot vector of the piece at that position
        # the 65th element will be the value of the position
        # the 66th element will be the value of the best move
        # doesn't need more than five, because the best move is always the first move
        path = os.path.join('stockfish_15.1_linux_x64_bmi2', 'stockfish_15.1_x64_bmi2')
        stockfish = Stockfish(path)

        # create a new data frame
        new_data = pd.DataFrame()
        new_data_list = []
        time_start = time.time()
        # process the data from FEN notation to a 2d array of 64 elements
        for each, frame  in data.iterrows():
            # get the FEN notation
            fen = frame[1]

            # get chess library board object
            chess_board = chess.Board(fen)

            # set stockfish to the position
            stockfish.set_fen_position(fen)

            # get a string containing the sequence of moves
            lichess_move = frame[2].split()[0]

            # get the best move according to stockfish
            stockfish_move = Stockfish.get_best_move(stockfish)

            # get the value of the position
            value = Stockfish.get_evaluation(stockfish)['value']

            # get the rating of the best move
            best_move_rating = frame[3]

            # get the 2d array of 64 elements
            board = self.get_board(fen)

            # get the 1-hot vector of the lichess move
            best_lichess_move_vector = self.get_best_move_vector(lichess_move)

            # get the 1-hot vector of the stockfish move
            best_stockfish_move_vector = self.get_best_move_vector(stockfish_move)

            # add the values above to new data frame
            # add the 2d array of 64 elements to the data frame
            # add the value of the position to the data frame
            # add the value of the best move to the data frame
            # add the 1-hot vector of the lichess move to the data frame
            # add the 1-hot vector of the stockfish move to the data frame
            new_data_list.append((board, value, best_move_rating, best_lichess_move_vector, best_stockfish_move_vector))

            if each % 10000 == 0:
                new_data = pd.DataFrame(new_data_list, columns=['board', 'value', 'best_move_rating', 'best_lichess_move_vector', 'best_stockfish_move_vector'])
                time_end = time.time()
                print('time taken: ', time_end - time_start)
                time_start = time.time()
                with open(self.save_file, 'wb') as f:
                    pickle.dump(new_data, f)


        new_data = pd.DataFrame(new_data_list, columns=['board', 'value', 'best_move_rating', 'best_lichess_move_vector', 'best_stockfish_move_vector'])

        with open(self.save_file, 'wb') as f:
            pickle.dump(new_data, f)


        return new_data

    def get_board(self, fen):
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
    
    def get_best_move_vector(self, move):
        # get the 1-hot vector of the best move
        # the move will be in algebraic notation
        # the move will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]
        # promotion will be in the format of [queen, rook, bishop, knight]
        # capture will be in the format of [capture, no capture]
        # check will be in the format of [check, no check]
        # checkmate will be in the format of [checkmate, no checkmate]
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]

        # get the from and to squares
        from_square = move[:2]
        to_square = move[2:4]

        # get the promotion piece
        promotion = move[4:]

        # get the 1-hot vector of the from square
        from_square_vector = self.get_square_vector(from_square)

        # get the 1-hot vector of the to square
        to_square_vector = self.get_square_vector(to_square)

        # get the 1-hot vector of the promotion piece
        promotion_vector = self.get_promotion_vector(promotion)

        # get the 1-hot vector of the capture
        capture_vector = self.get_capture_vector(move)

        # get the 1-hot vector of the check
        check_vector = self.get_check_vector(move)

        # get the 1-hot vector of the checkmate
        checkmate_vector = self.get_checkmate_vector(move)

        # get the 1-hot vector of the best move
        best_move_vector = np.concatenate((from_square_vector, to_square_vector, promotion_vector, capture_vector, check_vector, checkmate_vector))

        return best_move_vector
    
    def get_square_vector(self, square):
        # get the 1-hot vector of the square
        # the square will be in algebraic notation
        # the square will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [file, rank]
        # the 1-hot vector will be in the format of [file, rank]

        # get the file and rank
        file = square[0]
        rank = square[1]

        # get the 1-hot vector of the file
        file_vector = self.get_file_vector(file)

        # get the 1-hot vector of the rank
        rank_vector = self.get_rank_vector(rank)

        # get the 1-hot vector of the square
        square_vector = np.concatenate((file_vector, rank_vector))

        return square_vector
    
    def get_file_vector(self, file):
        # get the 1-hot vector of the file
        # the file will be in algebraic notation
        # the file will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [file, rank]

        # get the 1-hot vector of the file
        file_vector = np.zeros(8)
        file_vector[ord(file) - 97] = 1

        return file_vector
    
    def get_rank_vector(self, rank):
        # get the 1-hot vector of the rank
        # the rank will be in algebraic notation
        # the rank will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [file, rank]

        # get the 1-hot vector of the rank
        rank_vector = np.zeros(8)
        rank_vector[int(rank) - 1] = 1

        return rank_vector
    
    def get_promotion_vector(self, promotion):
        # get the 1-hot vector of the promotion piece
        # the promotion piece will be in algebraic notation
        # the promotion piece will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [queen, rook, bishop, knight]

        # get the 1-hot vector of the promotion piece
        promotion_vector = np.zeros(4)
        if promotion == 'q':
            promotion_vector[0] = 1
        elif promotion == 'r':
            promotion_vector[1] = 1
        elif promotion == 'b':
            promotion_vector[2] = 1
        elif promotion == 'n':
            promotion_vector[3] = 1

        return promotion_vector
    
    def get_capture_vector(self, move):
        # get the 1-hot vector of the capture
        # the capture will be in algebraic notation
        # the capture will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [capture, no capture]

        # get the 1-hot vector of the capture
        capture_vector = np.zeros(2)
        if 'x' in move:
            capture_vector[0] = 1
        else:
            capture_vector[1] = 1

        return capture_vector
    
    def get_check_vector(self, move):
        # get the 1-hot vector of the check
        # the check will be in algebraic notation
        # the check will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [check, no check]

        # get the 1-hot vector of the check
        check_vector = np.zeros(2)
        if '+' in move:
            check_vector[0] = 1
        else:
            check_vector[1] = 1

        return check_vector
    
    def get_checkmate_vector(self, move):
        # get the 1-hot vector of the checkmate
        # the checkmate will be in algebraic notation
        # the checkmate will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [checkmate, no checkmate]

        # get the 1-hot vector of the checkmate
        checkmate_vector = np.zeros(2)
        if '#' in move:
            checkmate_vector[0] = 1
        else:
            checkmate_vector[1] = 1

        return checkmate_vector
    
    def get_best_move(self, board):
        # get the best move
        # the best move will be in algebraic notation
        # the best move will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]
        # promotion will be in the format of [queen, rook, bishop, knight]
        # capture will be in the format of [capture, no capture]
        # check will be in the format of [check, no check]
        # checkmate will be in the format of [checkmate, no checkmate]
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]

        # get the best move
        best_move = board.best_move()

        # get the 1-hot vector of the best move
        best_move_vector = self.get_best_move_vector(best_move)

        return best_move_vector
    

class ChessBoard:
    def __init__(self, board):
        # initialize the chess board
        # the chess board will be in algebraic notation
        # the chess board will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [white, black]
        # the 1-hot vector will be in the format of [white, black]

        # get the 1-hot vector of the chess board
        board_vector = self.get_board_vector(board)

        # set the 1-hot vector of the chess board
        self.board_vector = board_vector

    def get_board_vector(self, board):
        # get the 1-hot vector of the chess board
        # the chess board will be in algebraic notation
        # the chess board will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [white, black]
        # the 1-hot vector will be in the format of [white, black]

        # get the 1-hot vector of the chess board
        board_vector = np.zeros(2)
        if board.turn:
            board_vector[0] = 1
        else:
            board_vector[1] = 1

        return board_vector
    

class ChessMove:
    def __init__(self, move):
        # initialize the chess move
        # the chess move will be in algebraic notation
        # the chess move will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]
        # promotion will be in the format of [queen, rook, bishop, knight]
        # capture will be in the format of [capture, no capture]
        # check will be in the format of [check, no check]
        # checkmate will be in the format of [checkmate, no checkmate]
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]

        # get the 1-hot vector of the chess move
        move_vector = self.get_move_vector(move)

        # set the 1-hot vector of the chess move
        self.move_vector = move_vector

    def get_move_vector(self, move):
        # get the 1-hot vector of the chess move
        # the chess move will be in algebraic notation
        # the chess move will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]
        # promotion will be in the format of [queen, rook, bishop, knight]
        # capture will be in the format of [capture, no capture]
        # check will be in the format of [check, no check]
        # checkmate will be in the format of [checkmate, no checkmate]
        # the 1-hot vector will be in the format of [from, to, promotion, capture, check, checkmate]
        # from and to will be in the format of [file, rank]

        # get the 1-hot vector of the chess move
        move_vector = np.zeros(6)

        # get the 1-hot vector of the from square
        from_square_vector = self.get_from_square_vector(move)

        # get the 1-hot vector of the
        to_square_vector = self.get_to_square_vector(move)

        # get the 1-hot vector of the promotion
        promotion_vector = self.get_promotion_vector(move)

        # get the 1-hot vector of the capture
        capture_vector = self.get_capture_vector(move)

        # get the 1-hot vector of the check
        check_vector = self.get_check_vector(move)

        # get the 1-hot vector of the checkmate
        checkmate_vector = self.get_checkmate_vector(move)

        # set the 1-hot vector of the chess move
        move_vector[0] = from_square_vector[0]
        move_vector[1] = from_square_vector[1]
        move_vector[2] = to_square_vector[0]
        move_vector[3] = to_square_vector[1]
        move_vector[4] = promotion_vector[0]
        move_vector[5] = promotion_vector[1]
        move_vector[6] = promotion_vector[2]
        move_vector[7] = promotion_vector[3]
        move_vector[8] = capture_vector[0]
        move_vector[9] = capture_vector[1]
        move_vector[10] = check_vector[0]
        move_vector[11] = check_vector[1]
        move_vector[12] = checkmate_vector[0]
        move_vector[13] = checkmate_vector[1]
        
        return move_vector
    
    def get_from_square_vector(self, move):
        # get the 1-hot vector of the from square
        # the from square will be in algebraic notation
        # the from square will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [file, rank]
        # the 1-hot vector will be in the format of [file, rank]

        # get the 1-hot vector of the from square
        from_square_vector = np.zeros(2)
        from_square_vector[0] = move.from_square % 8
        from_square_vector[1] = move.from_square // 8

        return from_square_vector
    
    def get_to_square_vector(self, move):
        # get the 1-hot vector of the to square
        # the to square will be in algebraic notation
        # the to square will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [file, rank]
        # the 1-hot vector will be in the format of [file, rank]

        # get the 1-hot vector of the to square
        to_square_vector = np.zeros(2)
        to_square_vector[0] = move.to_square % 8
        to_square_vector[1] = move.to_square // 8

        return to_square_vector
    
    def get_promotion_vector(self, move):
        # get the 1-hot vector of the promotion
        # the promotion will be in algebraic notation
        # the promotion will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [queen, rook, bishop, knight]
        # the 1-hot vector will be in the format of [queen, rook, bishop, knight]

        # get the 1-hot vector of the promotion
        promotion_vector = np.zeros(4)
        if move.promotion == chess.QUEEN:
            promotion_vector[0] = 1
        elif move.promotion == chess.ROOK:
            promotion_vector[1] = 1
        elif move.promotion == chess.BISHOP:
            promotion_vector[2] = 1
        elif move.promotion == chess.KNIGHT:
            promotion_vector[3] = 1

        return promotion_vector
    
    def get_capture_vector(self, move):
        # get the 1-hot vector of the capture
        # the capture will be in algebraic notation
        # the capture will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [capture, no capture]
        # the 1-hot vector will be in the format of [capture, no capture]

        # get the 1-hot vector of the capture
        capture_vector = np.zeros(2)
        if move.captured_piece_type:
            capture_vector[0] = 1
        else:
            capture_vector[1] = 1

        return capture_vector
    
    def get_check_vector(self, move):
        # get the 1-hot vector of the check
        # the check will be in algebraic notation
        # the check will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [check, no check]
        # the 1-hot vector will be in the format of [check, no check]

        # get the 1-hot vector of the check
        check_vector = np.zeros(2)
        if move.is_check():
            check_vector[0] = 1
        else:
            check_vector[1] = 1

        return check_vector
    
    def get_checkmate_vector(self, move):
        # get the 1-hot vector of the checkmate
        # the checkmate will be in algebraic notation
        # the checkmate will be converted to a 1-hot vector
        # the 1-hot vector will be in the format of [checkmate, no checkmate]
        # the 1-hot vector will be in the format of [checkmate, no checkmate]

        # get the 1-hot vector of the checkmate
        checkmate_vector = np.zeros(2)
        if move.is_checkmate():
            checkmate_vector[0] = 1
        else:
            checkmate_vector[1] = 1

        return checkmate_vector
    
    def get_move_from_move_vector(self, move_vector):
        # get the chess move from the 1-hot vector of the chess move
        # the 1-hot vector of the chess move will be in the format of [from square, to square, promotion, capture, check, checkmate]
        # the 1-hot vector of the chess move will be in the format of [from square, to square, promotion, capture, check, checkmate]

        # get the from square from the 1-hot vector of the from square
        from_square = self.get_square_from_square_vector(move_vector[0:2])

        # get the to square from the 1-hot vector of the to square
        to_square = self.get_square_from_square_vector(move_vector[2:4])

        # get the promotion from the 1-hot vector of the promotion
        promotion = self.get_piece_from_piece_vector(move_vector[4:8])

        # get the capture from the 1-hot vector of the capture
        capture = self.get_piece_from_piece_vector(move_vector[8:10])

        # get the check from the 1-hot vector of the check
        check = self.get_piece_from_piece_vector(move_vector[10:12])

        # get the checkmate from the 1-hot vector of the checkmate
        checkmate = self.get_piece_from_piece_vector(move_vector[12:14])

        # get the chess move from the from square, to square, promotion, capture, check, and checkmate
        move = chess.Move(from_square, to_square, promotion, capture, check, checkmate)

        return move
    
    def get_square_from_square_vector(self, square_vector):
        # get the square from the 1-hot vector of the square
        # the 1-hot vector of the square will be in the format of [file, rank]
        # the 1-hot vector of the square will be in the format of [file, rank]

        # get the square from the 1-hot vector of the square
        square = square_vector[0] + 8 * square_vector[1]

        return square
    
    def get_piece_from_piece_vector(self, piece_vector):
        # get the piece from the 1-hot vector of the piece
        # the 1-hot vector of the piece will be in the format of [queen, rook, bishop, knight]
        # the 1-hot vector of the piece will be in the format of [queen, rook, bishop, knight]

        # get the piece from the 1-hot vector of the piece
        piece = None
        if piece_vector[0] == 1:
            piece = chess.QUEEN
        elif piece_vector[1] == 1:
            piece = chess.ROOK
        elif piece_vector[2] == 1:
            piece = chess.BISHOP
        elif piece_vector[3] == 1:
            piece = chess.KNIGHT

        return piece
    
# this file is only for processing the data
# other files will take the data and use it to train the neural network
# this file will not be used to train the neural network

# need to test the code above. It doesn't look to be actually saving the processed data to a file

# this file is only for processing the data













