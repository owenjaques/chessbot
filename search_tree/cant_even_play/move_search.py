# Monte carlo tree search for move selection in chess with python-chess and numpy (numpy is used for the board representation)
# Uses the UCB1 formula for move selection
# Generates complete tree to depth 4 (depth 5 is the root node)
# then rolls out a random game to completion from each leaf node
# and backpropagates the result to the root node
# the rollout should be replaced with a neural network evaluation that predicts moves from board states
# the neural network should be trained with supervised learning using the rollout results as the labels
# the neural network should be trained with self-play using the rollout results as the reward

# there should be classes for the tree search itself, the nodes, and the neural network

# the tree search should be able to be run in parallel on multiple boards with a single neural network

# the model should be saved and loaded from disk and train between sessions with time constraints

# the model should be able to be trained with supervised learning and self-play

import chess
import chess.pgn
import numpy as np
import random
import time
import math
import copy
import sys
import os
import multiprocessing
import threading
import queue

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


from keras.models import load_model

from keras.models import model_from_json

from keras.models import model_from_yaml

from keras.models import model_from_config


class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.untried_moves = list(board.legal_moves)
        self.player_just_moved = board.turn
        self.wins = 0
        self.visits = 0
        self.move = None

    def uct_select_child(self):
        s = sorted(self.children, key=lambda c: c.wins/c.visits + math.sqrt(2*math.log(self.visits)/c.visits))[-1]
        return s
    
    def add_child(self, m, board):
        n = Node(board, self)
        n.move = m
        self.untried_moves.remove(m)
        self.children.append(n)
        return n
    
    def update(self, result):
        self.visits += 1
        self.wins += result

    def rollout_policy(self, board):
        return random.choice(list(board.legal_moves))
    
    def rollout(self):
        board = self.board.copy()
        while not board.is_game_over():
            board.push(self.rollout_policy(board))
        if board.is_checkmate():
            if board.turn:
                return 0
            else:
                return 1
        else:
            return .5
        
    def is_terminal_node(self):
        return self.board.is_game_over()
    
    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untried_moves) + "]"
    
    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.children:
            s += c.tree_to_string(indent+1)
        return s
    
    def indent_string(self, indent):
        s = ""
        for i in range(1, indent+1):
            s += "   "
        return s
    
    def children_to_string(self):
        s = ""
        for c in self.children:
            s += str(c) + "\n"
        return s
    
class MCTS:
    def __init__(self, board, nn):
        self.root = Node(board)
        self.nn = nn
    
    def get_move(self, board, timeLimit=1):
        self.root = Node(board)
        start = time.time()
        while time.time() - start < timeLimit:
            v = self.tree_policy(self.root)
            reward = v.rollout()
            self.backpropagate(v, reward)
        return sorted(self.root.children, key=lambda c: c.visits)[-1].move
    
    def tree_policy(self, node):
        while not node.is_terminal_node():
            if len(node.untried_moves) > 0:
                return self.expand(node)
            else:
                node = node.uct_select_child()
        return node
    
    def expand(self, node):
        m = random.choice(node.untried_moves)
        board = node.board.copy()
        board.push(m)
        return node.add_child(m, board)
    
    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent
    
    def __repr__(self):
        return self.root.tree_to_string(0)
    
    def children_to_string(self):
        return self.root.children_to_string()
    

class NeuralNetwork:
    def __init__(self):
        if os.path.isfile('model.h5'):
            self.model = load_model('model.h5')
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), input_shape=(8, 8, 12), activation='relu'))
            self.model.add(Conv2D(32, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train(self, X, y, epochs=10, batch_size=200):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filename):
        self.model.save(filename)

# model isn't actually saved anywhere, so this is useless
    def save_to_yaml(self, filename):
        model_yaml = self.model.to_yaml()
        with open(filename, "w") as yaml_file:
            yaml_file.write(model_yaml)

# model isn't actually saved anywhere, so this is useless
    def save_to_json(self, filename):
        model_json = self.model.to_json()
        with open(filename, "w") as json_file:
            json_file.write(model_json)
    
    def load(self, filename):
        self.model = load_model(filename)
    
    def load_from_yaml(self, filename):
        yaml_file = open(filename, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
    
    def load_from_json(self, filename):
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
    
    def load_from_config(self, filename):
        self.model = model_from_config(filename)
    
    def load_from_h5(self, filename):
        self.model = load_model(filename)
    
    def load_from_weights(self, filename):
        self.model.load_weights(filename)


class ChessPlayer:
    def __init__(self, board, color, timeLimit=1):
        self.board = board
        self.color = color
        self.timeLimit = timeLimit
        self.nn = NeuralNetwork()
        self.mcts = MCTS(self.board, self.nn)
    
    def get_move(self):
        return self.mcts.get_move(self.board, self.timeLimit)
    
    def train(self, X, y, epochs=10, batch_size=200):
        self.nn.train(X, y, epochs, batch_size)
    
    def save(self, filename):
        self.nn.save(filename)
    
    def load(self, filename):
        self.nn.load(filename)
    
    def load_from_yaml(self, filename):
        self.nn.load_from_yaml(filename)
    
    def load_from_json(self, filename):
        self.nn.load_from_json(filename)
    
    def load_from_config(self, filename):
        self.nn.load_from_config(filename)
    
    def load_from_h5(self, filename):
        self.nn.load_from_h5(filename)
    
    def load_from_weights(self, filename):
        self.nn.load_from_weights(filename)
    
    def get_board(self):
        return self.board
    
    def set_board(self, board):
        self.board = board
        self.mcts = MCTS(self.board, self.nn)
    
    def get_color(self):
        return self.color
    
    def set_color(self, color):
        self.color = color
    
    def get_time_limit(self):
        return self.timeLimit
    
    def set_time_limit(self, timeLimit):
        self.timeLimit = timeLimit


class ChessGame:
    def __init__(self, white, black):
        self.white = white
        self.black = black
        self.board = chess.Board()
        self.white.set_board(self.board)
        self.black.set_board(self.board)
    
    def play_game(self):
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                move = self.white.get_move()
            else:
                move = self.black.get_move()
            self.board.push(move)
    
    def get_board(self):
        return self.board
    
    def get_white(self):
        return self.white
    
    def get_black(self):
        return self.black
    
    def set_white(self, white):
        self.white = white
    
    def set_black(self, black):
        self.black = black

def board_to_array(board):
    board_array = np.zeros((8, 8, 12))
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                if piece.color == chess.WHITE:
                    board_array[i][j][piece.piece_type - 1] = 1
                else:
                    board_array[i][j][piece.piece_type + 5] = 1
    return board_array

def train(white, black, games=100, epochs=10, batch_size=200):
    for i in range(games):
        game = ChessGame(white, black)
        game.play_game()
        board = game.get_board()
        X = []
        y = []
        while not board.is_game_over():
            X.append(board_to_array(board))
            move = board.pop()
            y.append(move)
        X = np.array(X)
        y = np.array(y)
        white.train(X, y, epochs, batch_size)
        black.train(X, y, epochs, batch_size)
    return white, black

def play(white, black, games=100):
    white_wins = 0
    black_wins = 0
    draws = 0
    for i in range(games):
        game = ChessGame(white, black)
        game.play_game()
        board = game.get_board()
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                black_wins += 1
            else:
                white_wins += 1
        else:
            draws += 1
    return white_wins, black_wins, draws


def main():
    white = ChessPlayer(chess.Board(), chess.WHITE)
    black = ChessPlayer(chess.Board(), chess.BLACK)
    white, black = train(white, black, 10, 10, 200)
    white_wins, black_wins, draws = play(white, black, 10)
    print('White wins: ' + str(white_wins))
    print('Black wins: ' + str(black_wins))
    print('Draws: ' + str(draws))

    if white_wins > black_wins:
        model = white
    else:
        model = black

    model.save('model.h5')
        

if __name__ == '__main__':
    main()

