# Count, sort, and plot the frequency of each move in a game (or a set of games).
#
# Path: move_frequency.py

# input: a numpy array of 1-hot vectors of length 64*2 + 64*64  + 1
# output: a numpy array of 64*64 + 1, where the first 64*64 elements are the
#         frequency of each move, and the last element is the total number of
#         moves in the game

import numpy as np
import matplotlib.pyplot as plt

def moveFrequency(data, board_length=128):
    # initialize frequency array
    frequency = np.zeros(64*64 + 1)

    # for each move in the game
    for i in range(len(data) - 1):
        # get the move
        move = data[i, board_length:]

        # get the index of the move
        index = np.argmax(move)

        # increment the frequency of the move
        frequency[index] += 1

    # set the last element to the total number of moves in the game
    # should be equal to the number of non-zero elements in the frequency array
    frequency[-1] = np.count_nonzero(frequency)

    return frequency

def plotFrequency(frequency):
    # plot the frequency of each move in a game
    # frequency is a numpy array of 64*64 + 1, where the first 64*64 elements
    # are the frequency of each move, and the last element is the total number
    # of moves in the game

    # get the total number of moves in the game
    total_moves = frequency[-1]

    # get the frequency of each move
    move_frequency = frequency[:-1]

    # get the index of each move
    move_index = np.arange(64*64)

    # sort the moves by frequency
    sorted_index = np.argsort(move_frequency)[::-1]

    # plot the frequency of each move
    plt.plot(move_index, move_frequency[sorted_index])
    plt.xlabel('Move Index')
    plt.ylabel('Frequency')
    plt.title('Move Frequency in a Game')
    plt.show()

    # plot the cumulative frequency of each move
    plt.plot(move_index, np.cumsum(move_frequency[sorted_index]) / total_moves)
    plt.xlabel('Move Index')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative Move Frequency in a Game')
    plt.show()



# get the frequency of moves from each square in a game (or a set of games) and plot it
def fromSquareFrequency(data, board_length=128):
    # initialize frequency array
    frequency = np.zeros(64)

    # for each move in the game
    for i in range(len(data) - 1):
        # get the move
        move = data[i, board_length:]

        # get the index of the move
        index = np.argmax(move)

        # get the square the move was made from
        square = index // 64

        # increment the frequency of the square
        frequency[square] += 1

    # plot the frequency of each square
    plt.plot(np.arange(64), frequency)
    plt.xlabel('From Square Index')
    plt.ylabel('Frequency')
    plt.title('From Square Frequency in a Game')
    plt.show()

    # plot the cumulative frequency of each square
    plt.plot(np.arange(64), np.cumsum(frequency) / np.sum(frequency))
    plt.xlabel('From Square Index')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative From Square Frequency in a Game')
    plt.show()

# get the frequency of moves to each square in a game (or a set of games) and plot it
def toSquareFrequency(data, board_length=128):
    # initialize frequency array
    frequency = np.zeros(64)

    # for each move in the game
    for i in range(len(data) - 1):
        # get the move
        move = data[i, board_length:]

        # get the index of the move
        index = np.argmax(move)

        # get the square the move was made to
        square = index % 64

        # increment the frequency of the square
        frequency[square] += 1

    # plot the frequency of each square
    plt.plot(np.arange(64), frequency)
    plt.xlabel('To Square Index')
    plt.ylabel('Frequency')
    plt.title('To Square Frequency in a Game')
    plt.show()

    # plot the cumulative frequency of each square
    plt.plot(np.arange(64), np.cumsum(frequency) / np.sum(frequency))
    plt.xlabel('To Square Index')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative To Square Frequency in a Game')
    plt.show()


def main():
    # load data npz file
    with open("puzzle_0to749999_unfiltered_compressed.npz", "rb") as f:
        data = np.load(f, allow_pickle=True)["arr_0"]

    # get the frequency of each move in a game
    frequency = moveFrequency(data)

    # plot the frequency of each move in a game
    plotFrequency(frequency)