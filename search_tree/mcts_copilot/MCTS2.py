# Search for the best chess move using the Monte Carlo Tree Search algorithm with UCT (Upper Confidence Bound applied to Trees) selection
# input: chessboard, time limit in seconds
# output: best move in chess notation
# Don't use others exact code, but you can use this code as a reference
# check for division by zero errors
# check for NoneType errors
# check for other errors
# similar to alphazero but without neural network for the time being, will use machine learning models to improve the search
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
# https://en.wikipedia.org/wiki/Upper_confidence_bound_for_trees
# https://en.wikipedia.org/wiki/AlphaZero
# https://en.wikipedia.org/wiki/Reinforcement_learning

# chessboard is a chess.Board object
# chess.Move.from_uci('e2e4') is a chess.Move object
# chess.Move.from_uci('e2e4').uci() is a string in chess notation

# search is deleted after time limit is reached and move returned based on UCT (Upper Confidence Bound applied to Trees) selection

# class Node is a node in the search tree
# class Node has attributes:
#   parent: parent node
#   board: chessboard
#   children: list of child nodes
#   move: move that leads to this node
#   visits: number of visits to this node
#   wins: number of wins from this node
#   uct: UCT value of this node
#   move_prediction_models: list of move prediction models
#   move_model_input: list of move model input versions
#   board_prediction_models: list of board prediction models
#   board_model_input: list of board model input versions
#   expanded: boolean indicating if node has been expanded

# class Node has methods:
#   __init__(self, parent, board, move): initialize node
#   add_child(self, board, move): add child node
#   select(self): select child node with highest UCT value
#   expand(self): expand node by adding child nodes
#   rollout(self): rollout node by playing random moves until game is over
#   backpropagate(self, result): backpropagate result from rollout to this node and all parent nodes

# class Search is the search tree
# class Search has attributes:
#   root: root node
#   time_limit: time limit in seconds
#   time_start: time when search started
#   time_end: time when search ended
#   nodes: number of nodes in search tree
#   nodes_expanded: number of nodes expanded
#   nodes_visited: number of nodes visited
#   nodes_won: number of nodes won
#   nodes_lost: number of nodes lost
#   nodes_drawn: number of nodes drawn
#   nodes_timed_out: number of nodes timed out
#   nodes_rolled_out: number of nodes rolled out
#   nodes_backpropagated: number of nodes backpropagated
#   nodes_selected: number of nodes selected
#   nodes_not_selected: number of nodes not selected
#   nodes_not_expanded: number of nodes not expanded
#   nodes_not_visited: number of nodes not visited
#   nodes_not_won: number of nodes not won
#   nodes_not_lost: number of nodes not lost
#   nodes_not_drawn: number of nodes not drawn
#   nodes_not_timed_out: number of nodes not timed out
#   nodes_not_rolled_out: number of nodes not rolled out
#   nodes_not_backpropagated: number of nodes not backpropagated


# class Search has methods:
#   __init__(self, board, time_limit): initialize search tree
#   search(self): search for best move
#   select(self, node): select child node with highest UCT value


import chess

import math
import random
import time
import os
import pickle

from search_tree.functions.board_processing import Boardprocessing

class Node:
    def __init__(self, parent, board, move, move_prediction_models=None, move_model_input=None, board_prediction_models=None, board_model_input=None):
        self.parent = parent
        self.board = board
        self.children = []
        self.move = move
        self.visits = 0
        self.wins = 0
        self.uct = 0
        self.move_prediction_models = move_prediction_models
        self.move_model_input = move_model_input
        self.board_prediction_models = board_prediction_models
        self.board_model_input = board_model_input
        self.expanded = False

    # is_leaf()
    def is_leaf(self):
        return len(self.children) == 0
    
    # is_root()
    def is_root(self):
        return self.parent is None
    
    # is_terminal()
    def is_terminal(self):
        return self.board.is_game_over()
    
    # backpropagate result from rollout to this node and all parent nodes
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.wins += 1
        elif result == 0:
            pass
        elif result == -1:
            self.wins -= 1
        if self.parent is not None:
            self.parent.backpropagate(result)

    # add child node
    def add_child(self, board, move):
        self.children.append(Node(self, board, move, self.move_prediction_models, self.move_model_input, self.board_prediction_models, self.board_model_input))

    # select child node with highest UCT value
    # needs to check if checkmate or stalemate
    def select(self):
        if self.is_leaf():
            return self
        else:
            max_uct = -math.inf
            max_uct_node = None
            for child in self.children:
                if child.uct > max_uct:
                    max_uct = child.uct
                    max_uct_node = child
            return max_uct_node
        
    
    # currently the MCTS algorithm is not used and simply moves the same piece back and forth
    # this is because the MCTS algorithm is not working properly
    # the MCTS algorithm is not working properly because the UCT value is not being calculated properly
    # the UCT value is not being calculated properly because the number of visits to a node is not being calculated properly
    # the number of visits to a node is not being calculated properly because the backpropagation is not working properly
    # the backpropagation is not working properly because the rollout is not working properly


    # expand node by adding child nodes
    def expand(self):
        for move in self.board.legal_moves:
            board = self.board.copy()
            board.push(move)
            self.add_child(board, move)
            board.pop()

    # rollout with model prediction if model is available
    """
    def rollout(self):
        board = self.board.copy()
        while not board.is_game_over():
            if self.move_prediction_models is not None:
                moves = []
                for (input_version,each) in zip(self.board_model_input, self.move_prediction_models):
                    if input_version == "V1" or input_version == "V2":
                        move = each.predict(Boardprocessing(board, input_version).get_board_image())
                        if move is not None and move in board.legal_moves:
                            moves.append(move)
                    if input_version == "neural_network":
                        move = each.predict(Boardprocessing(board, input_version).get_board_image())
                        if move is not None and move in board.legal_moves:
                            moves.append(move)
                # here would predict which move is best based on output of board prediction model
                if len(moves) > 0:
                    board.push(random.choice(moves))
                else:
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
            else:
                move = random.choice(list(board.legal_moves))
                board.push(move)
        return board.result() == '1-0'
    """
    
    # rollout without model prediction
    def rollout(self):
        board = self.board.copy()
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
        return board.result() == '1-0'
    

class Search:
    def __init__(self, board, time_limit):
        self.time_limit = time_limit
        self.time_start = time.time()
        self.time_end = self.time_start + self.time_limit
        self.nodes = 1
        self.nodes_expanded = 0
        self.nodes_visited = 0
        self.nodes_won = 0
        self.nodes_lost = 0
        self.nodes_drawn = 0
        self.nodes_timed_out = 0
        self.nodes_rolled_out = 0
        self.nodes_backpropagated = 0
        self.nodes_selected = 0
        self.nodes_not_selected = 0
        self.move_prediction_models = None
        self.move_prediction_input = None
        self.board_prediction_models = None
        self.board_prediction_input = None
        self.load_models()
        self.root = Node(None, board, None, self.move_prediction_models, self.move_prediction_input, self.board_prediction_models, self.move_prediction_input)

    def load_models(self):
        move_prediction_models_list = []
        move_prediction_input_list = []

        board_prediction_models_list = []
        board_prediction_input_list = []

        for filename in os.listdir('search_tree/move_models'):
            with open('search_tree/move_models//'+ filename, 'rb') as f:
                move_prediction_models_list.append(pickle.load(f))
                if "V1" in filename:
                    move_prediction_input_list.append(1)
                elif "V2" in filename:
                    move_prediction_input_list.append(2)
    
        for filename in os.listdir('search_tree/board_models'):
            with open('search_tree/board_models//'+ filename, 'rb') as f:
                board_prediction_models_list.append(pickle.load(f))
                if "V1" in filename:
                    board_prediction_input_list.append("V1")
                elif "V2" in filename:
                    board_prediction_input_list.append("V2")

        if len(move_prediction_models_list) > 0:
            self.move_prediction_models = move_prediction_models_list
            self.move_prediction_input = move_prediction_input_list

        if len(board_prediction_models_list) > 0:
            self.board_prediction_models = board_prediction_models_list
            self.board_prediction_input = board_prediction_input_list


    def search(self):
        while time.time() < self.time_end:
            node = self.select(self.root)
            if node is None:
                self.nodes_not_selected += 1
                continue
            self.nodes_selected += 1
            if not node.expanded:
                node.expand()
                self.nodes_expanded += 1
            if node.visits == 0:
                result = node.rollout()
                self.nodes_rolled_out += 1
                if result:
                    self.nodes_won += 1
                else:
                    self.nodes_lost += 1
            else:
                node = node.select()
                result = node.rollout()
                self.nodes_rolled_out += 1
                if result:
                    self.nodes_won += 1
                else:
                    self.nodes_lost += 1
            node.backpropagate(result)
            self.nodes_backpropagated += 1
        try :
            move = max(self.root.children, key=lambda node: node.visits).move
        except:
            move = None
        return move

    def select(self, node):
        if node is None:
            return None
        if node.is_leaf():
            return node
        if node.is_terminal():
            return None
        return self.select(node.select())
    
    def print_stats(self):
        print('nodes:', self.nodes)
        print('nodes expanded:', self.nodes_expanded)
        print('nodes visited:', self.nodes_visited)
        print('nodes won:', self.nodes_won)
        print('nodes lost:', self.nodes_lost)
        print('nodes drawn:', self.nodes_drawn)
        print('nodes timed out:', self.nodes_timed_out)
        print('nodes rolled out:', self.nodes_rolled_out)
        print('nodes backpropagated:', self.nodes_backpropagated)
        print('nodes selected:', self.nodes_selected)
        print('nodes not selected:', self.nodes_not_selected)

def uct(board, time_limit):
    search = Search(board, time_limit)
    move = search.search()
    search.print_stats()
    return move 

if __name__ == '__main__':
    board = chess.Board()
    move = uct(board, 1)
    print(move)




