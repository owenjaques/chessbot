# input: chess fen string
# output: a numpy array of length max_moves, where each element is a 1-hot vector if the move at index is a legal move, and 0 otherwise
#


import numpy as np
import chess

def encode_legal_moves(board):
    # encode the legal moves of a board as a numpy array of 1-hot vectors
    # board is a chess board
    # returns a numpy array of 1-hot vectors of length 64*64

    # initialize legal_moves
    legal_moves = np.zeros((64*64))

    # for each move in the board
    for i, move in enumerate(board.legal_moves):
        # get the index of the move
        index = move.from_square * 64 + move.to_square

        # set the index of the move to 1
        legal_moves[i, index] = 1

    return legal_moves

def encode_move_uci(move):
    # encode chess move as a 1-hot vector of length 64*64
    # move is a string in the form 'e2e4'
    # returns a 1-hot vector of length 64*64

    # convert move to a tuple of ints
    move = (int(move[1]) - 1) * 8 + ord(move[0]) - 97, (int(move[3]) - 1) * 8 + ord(move[2]) - 97

    # encode move as a 1-hot vector
    move = np.eye(64*64, dtype=np.int8)[move[0]*64 + move[1]]

    return move

def decode_move_uci(move):
    # decode a 1-hot vector of length 64*64 to a chess move
    # move is a 1-hot vector of length 64*64
    # returns a string in the form 'e2e4'

    # get the index of the 1 in the vector
    index = int(np.argmax(move))

    # convert the index to a move
    move =  chr((index // 64) % 8 + 97) + str((index // 64) // 8 + 1) + chr((index % 64) % 8 + 97) + str((index % 64) // 8 + 1)

    return move