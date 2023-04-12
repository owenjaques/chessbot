# for each element in the data, make the first move and update the board, if there is no more moves, remove the element from the data
#

# input: csv file with puzzle id, fen, and moves
# output: csv file with puzzle id, fen after first move, and moves without first move

# import libraries
import pandas as pd
import numpy as np
import time
import chess
import chess.pgn
import chess.svg
import os
import sys


# no need to split the data into chunks, just make the first move and save the data
def make_move(data, save_file=None):
    # convert the data to a numpy array
    data = np.array(data)

    for i, each in enumerate(data):
        if i % 10000 == 0:
            print('Processing row ' + str(i) + ' of ' + str(len(data)) + '.')
        # make the first move
        board = chess.Board(each[1])
        board.push_san(each[2].split()[0])
        data[i][1] = board.fen()
        data[i][2] = ' '.join(each[2].split()[1:])

    # remove the rows with no moves
    data = data[data[:, 2] != '']
    data = data[data[:, 2] != ' ']

    # convert the data back to a pandas dataframe
    data = pd.DataFrame(data)

    # save the data to a csv file
    data.to_csv('data_raw/lichess_db_puzzle_move_8.csv', header=None, index=None)


# main function
def main():
    # load the data
    data = pd.read_csv('data_raw/lichess_db_puzzle_move_7.csv', header=None)

    # make the first move
    make_move(data)


# run the main function
if __name__ == '__main__':
    main()

