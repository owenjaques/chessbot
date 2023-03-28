# Search for the best chess move using the Monte Carlo Tree Search algorithm with UCT (Upper Confidence Bound applied to Trees) selection
# input: chessboard, time limit in seconds
# output: best move in chess notation
# Don't use others exact code, but you can use this code as a reference
# similar to alphazero but without neural network for the time being, will use machine learning models to improve the search
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
# https://en.wikipedia.org/wiki/Upper_confidence_bound_for_trees
# https://en.wikipedia.org/wiki/AlphaZero
# https://en.wikipedia.org/wiki/Reinforcement_learning

# chessboard is a chess.Board object
# chess.Move.from_uci('e2e4') is a chess.Move object
# chess.Move.from_uci('e2e4').uci() is a string in chess notation

# search is deleted after time limit is reached and move returned based on UCT (Upper Confidence Bound applied to Trees) selection
# return best move in chess notation after time limit is reached

# class Node is a node in the search tree
# class Node has attributes:
#   parent: parent node
#   board: chessboard
#   move: move to get to this node
#   children: list of child nodes
#   visits: number of visits to this node
#   wins: number of wins for this node
#   losses: number of losses for this node
#   draws: number of draws for this node
#   timed_out: number of times this node timed out
#   rolled_out: number of times this node was rolled out
#   backpropagated: number of times this node was backpropagated
#   selected: number of times this node was selected
#   not_selected: number of times this node was not selected


# class Node has methods:
#   __init__(self, parent, board, move): initialize node
#   expand(self): expand node by adding child nodes
#   rollout(self): rollout node by playing random moves until game is over
#   backpropagate(self, result): backpropagate result to parent nodes
#   select(self): select child node with highest UCT value


# class Search is the search tree
# class Search has attributes:
#   root: root node
#   time_limit: time limit in seconds
#   start_time: time when search started
#   end_time: time when search ended
#   nodes: number of nodes in search tree
#   expanded: number of nodes expanded
#   rolled_out: number of nodes rolled out
#   backpropagated: number of nodes backpropagated
#   selected: number of nodes selected
#   not_selected: number of nodes not selected
#   timed_out: number of nodes timed out
#   wins: number of wins
#   losses: number of losses
#   draws: number of draws

# class Search has methods:
#   __init__(self, chessboard, time_limit): initialize search tree
#   search(self): search for best move
#   get_best_move(self): get best move based on UCT (Upper Confidence Bound applied to Trees) selection

import chess
import math
import random
import time

import os
import pickle

from search_tree.functions.board_processing import Boardprocessing


# class Node is a node in the search tree
class Node:
    # initialize node
    def __init__(self, parent, board, move, model=None):
        self.parent = parent
        self.board = board
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.timed_out = 0
        self.rolled_out = 0
        self.backpropagated = 0
        self.selected = 0
        self.not_selected = 0
        self.expanded = 0
        self.model = model

    # expand node by adding child nodes
    def expand(self):
        # get legal moves
        legal_moves = self.board.legal_moves
        # add child nodes
        for move in legal_moves:
            # make move
            self.board.push(move)
            # add child node
            self.children.append(Node(self, self.board.copy(), move, self.model))
            # undo move
            self.board.pop()

    # rollout node by playing random moves until game is over
    def rollout(self):
        # copy board
        board = self.board.copy()
        # play random moves until game is over
        while not board.is_game_over():
            # get legal moves
            legal_moves = board.legal_moves
            # get random move
            move = random.choice(list(legal_moves))
            # make move
            board.push(move)
        # get result
        if board.is_checkmate():
            if board.turn:
                result = 0
            else:
                result = 1
        elif board.is_stalemate():
            result = 0.5
        elif board.is_insufficient_material():
            result = 0.5
        elif board.can_claim_fifty_moves():
            result = 0.5
        elif board.can_claim_threefold_repetition():
            result = 0.5
        else:
            result = 0.5
        # return result
        return result
    
    # rollout remade with a logistic function model to select moves based on the model
    # model is found in the move_models folder and is called move_prediction_logreg_V1, import with pickle
    def rollout_logreg(self):
        # copy board
        board = self.board.copy()
        # play random moves until game is over
        while not board.is_game_over():
            # get legal moves
            legal_moves = board.legal_moves

            try:
                model_input = Boardprocessing(board).get_board_image()
                model_prediction = board.parse_san(self.model.predict(model_input)[0] )
                move = model_prediction
            except:
                move = random.choice(list(legal_moves))    
            # make move
            board.push(move)
        # get result
        if board.is_checkmate():
            if board.turn:
                result = 0
            else:
                result = 1
        elif board.is_stalemate():
            result = 0.5
        elif board.is_insufficient_material():
            result = 0.5
        elif board.can_claim_fifty_moves():
            result = 0.5
        elif board.can_claim_threefold_repetition():
            result = 0.5
        else:
            result = 0.5
        # return result
        return result


    # backpropagate result to parent nodes
    def backpropagate(self, result):
        # update visits
        self.visits += 1
        # update wins, losses, and draws
        if result == 1:
            self.wins += 1
        elif result == 0:
            self.losses += 1
        else:
            self.draws += 1
        # if parent node exists
        if self.parent:
            # backpropagate result to parent node
            self.parent.backpropagate(result)

    # select child node with highest UCT value
    def select(self):
        # get child nodes with highest UCT value
        best_child_nodes = []
        best_uct_value = -1
        for child_node in self.children:
            # get UCT value 
            # check for division by zero
            if child_node.visits == 0:
                uct_value = 0
            else:
                uct_value = (child_node.wins / child_node.visits) + (math.sqrt(2 * math.log(self.visits) / child_node.visits))
            # if UCT value is higher than best UCT value
            if uct_value > best_uct_value:
                # update best UCT value
                best_uct_value = uct_value
                # clear best child nodes
                best_child_nodes = []
                # add child node to best child nodes
                best_child_nodes.append(child_node)
            # if UCT value is equal to best UCT value
            elif uct_value == best_uct_value:
                # add child node to best child nodes
                best_child_nodes.append(child_node)
        # get random child node from best child nodes
        child_node = random.choice(best_child_nodes)
        # return child node
        return child_node
    
# class Search is the search tree
class Search:
    # initialize search tree
    def __init__(self, chessboard, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()
        self.end_time = self.start_time + self.time_limit
        self.nodes = 1
        self.expanded = 0
        self.rolled_out = 0
        self.backpropagated = 0
        self.selected = 0
        self.not_selected = 0
        self.timed_out = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.model = pickle.load(open('search_tree/move_models/move_prediction_logreg_V1.pkl', 'rb'))
        self.root = Node(None, chessboard, None, self.model)

    # search for best move
    def search(self):
        # while time limit has not been reached
        while time.time() < self.end_time:
            # select node
            node = self.root
            # while node has children
            while node.children:
                # select child node
                node = node.select()
                # update selected
                node.selected += 1
                # update not selected
                for child_node in node.parent.children:
                    if child_node != node:
                        child_node.not_selected += 1
            # expand node
            node.expand()
            # update nodes
            self.nodes += len(node.children)
            # update expanded
            node.expanded += 1
            # update expanded
            self.expanded += 1
            # if node has children
            if node.children:
                # select child node
                node = node.select()
                # update selected
                node.selected += 1
                # update not selected
                for child_node in node.parent.children:
                    if child_node != node:
                        child_node.not_selected += 1
            # rollout node
            result = node.rollout()
            #result = node.rollout_logreg()
            # update rolled out
            node.rolled_out += 1
            # update rolled out
            self.rolled_out += 1
            # update wins, losses, and draws
            if result == 1:
                self.wins += 1
            elif result == 0:
                self.losses += 1
            else:
                self.draws += 1
            # backpropagate result
            node.backpropagate(result)
            # update backpropagated
            node.backpropagated += 1
            # update backpropagated
            self.backpropagated += 1
        # get child nodes with highest visits
        best_child_nodes = []
        best_visits = -1
        for child_node in self.root.children:
            # if visits is higher than best visits
            if child_node.visits > best_visits:
                # update best visits
                best_visits = child_node.visits
                # clear best child nodes
                best_child_nodes = []
                # add child node to best child nodes
                best_child_nodes.append(child_node)
            # if visits is equal to best visits
            elif child_node.visits == best_visits:
                # add child node to best child nodes
                best_child_nodes.append(child_node)
        # get random child node from best child nodes
        best_child_node = random.choice(best_child_nodes)
        # return best move
        return best_child_node.move
    

# main
if __name__ == '__main__':
    # initialize chessboard
    chessboard = chess.Board()
    # initialize search tree
    search = Search(chessboard, 1)
    # search for best move
    best_move = search.search()
    # print best move
    print(best_move)

