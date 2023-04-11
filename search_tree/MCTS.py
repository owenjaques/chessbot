# MCTS for chess game using python-chess library
# shouldn't copy code from Github
#
# should use a neural network to evaluate the board

# https://joshvarty.github.io/AlphaZero/
# https://stackoverflow.com/questions/54027861/using-queue-priorityqueue-not-caring-about-comparisons
# https://web.stanford.edu/~surag/posts/alphazero.html


import chess
import random
import math
import time
import numpy as np
import copy
import os
import pickle
import queue
import heapq
import sys


import contextlib

from collections import defaultdict

sys.path.append('..')

from neural_networks.chessbot.chessbot import ChessBot
from neural_networks.chessbot.modelinput import ModelInput

from search_tree.experiments.CNN.board_processing import Boardprocessing


# have pretty heavily deviated from the original MCTS implementation
# is more of a UCT implementation now mixed with a few other ideas
class MCTS():
    def __init__(self, max_time=10, num_simulations=2500, color=chess.WHITE, max_depth=50, policy_nn=None, value_nn=None, value_nn_2=None, model_input=None, use_heap=False, expand_mode=False):
        self.board = chess.Board()
        self.player_color = color
        self.time_limit = max_time
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.policy = policy_nn
        self.value = value_nn
        self.value_two = value_nn_2
        self.heap_mark = use_heap
        self.nodes = {}
        self.best_move_value = 0
        self.model_input = model_input
        self.move_count = 0
        self.expand_mode = expand_mode
        self.depth = 0


    def get_best_move_value(self, board):
        # get the best move after the last search and return the value of the node
        move = self.search(board)
        return self.best_move_value, move

    def set_root(self, board):
        self.root = board.fen()
        self.move_count += 1
        if self.root not in self.nodes:
            self.nodes = {}
            self.nodes[self.root] = Node()
            self.nodes[self.root].set_board(self.root)
            self.nodes[self.root].add_visit(0)
            self.nodes[self.root].set_value(0)
            self.nodes[self.root].set_parent(None)
            self.nodes[self.root].set_action(None)
            if self.heap_mark:
                self.leaf_heapq = []
                heapq.heapify(self.leaf_heapq)
            self.expand( self.root )
        else:
            self.depth = self.nodes[self.root].depth
            self.nodes[self.root].parent = None
            # this isn't great yet... resets the heap every time
            # not sure how to efficiently update the heap yet as need to remove all nodes not connected to new root
            if self.heap_mark:
                self.leaf_heapq = [self.nodes[self.root]]
                heapq.heapify(self.leaf_heapq)

        
        
    def search(self, board):
        start_time = time.time()

        if len(list(board.legal_moves)) == 1:
            return list(board.legal_moves)[0]
        
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate() or board.result() == "1-0" or board.result() == "0-1":
                return move
            board.pop()

        # set the root node
        self.set_root(board)

        node = self.nodes[self.root]

        # while the time limit has not been reached
        while time.time() - start_time < self.time_limit:
            # get the next node to simulate
            if self.heap_mark and len(self.leaf_heapq) > 0:
                node = heapq.heappop(self.leaf_heapq).board.fen()
            else:
                node = self.root
                while not self.nodes[node].terminal:
                    node = self.select(node)
                    if node == None:
                        break

            if node != None:
                # expand the node
                if self.nodes[node].depth - self.depth < self.max_depth:
                    self.expand(node)
                    board = chess.Board(self.nodes[node].board.fen())
                    # left as a placeholder for now to test
                    value = 0
                    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
                        value = 1
                    elif board.result() == "1-0" or board.result() == "0-1" or board.is_checkmate():
                        value = -1.5
                    if value == 0:
                        if self.expand_mode:
                            value = self.evaluate(node)
                        else:
                            if self.value != None:
                                value = max(self.nodes[child].value for child in self.nodes[node].children)
                            else:
                                # unsure about this... need to test more
                                if self.heap_mark:
                                    value = sum(self.nodes[child].value for child in self.nodes[node].children)
                                    #value = self.evaluate(node)
                                    #value = self.evaluate(node)
                                    #value = self.rollout(node)
                                else:
                                    #value = sum(self.nodes[child].value for child in self.nodes[node].children)/len(self.nodes[node].children)
                                    #value = self.rollout(node)
                                    value = self.evaluate(node)
                else:
                    self.nodes[node].value = 0
                    value = 0

                self.backpropagate(node, value)

        # return the best move
        return self.best_move()
    

    def select(self, node):
        # select the best child of the node
        # if the node has no children, return None
        if len(self.nodes[node].children) == 0:
            return None
        else:
            min_value = math.inf
            best_child = None
            for child in self.nodes[node].children:
                child_value = self.nodes[child].value / self.nodes[child].visits - math.sqrt(2 * math.log(self.nodes[node].visits) / (self.nodes[child].visits+1))
                if child_value < min_value:
                    min_value = self.nodes[child].value
                    best_child = child
            return best_child

    
    def expand(self, node):
        # expand the node and add the children to the heap
        board_start = chess.Board(self.nodes[node].board.fen())

        if board_start != None:
            # to do: add a policy network to evaluate the board
            legal_moves = list(board_start.legal_moves)
            if len(legal_moves) == 0:
                self.nodes[node].terminal = True
                return

            for move in legal_moves:
                board = chess.Board(board_start.fen())
                board.push(move)
                child = Node()
                child.set_board(board.fen())
                child.set_parent(self.nodes[node].board.fen())
                child.add_visit(1)
                child.set_action(move)
                child.set_depth(self.nodes[node].depth + 1)
                if self.value != None:
                    if self.value_two != None:
                        # should be - if good for parent, + if good for opponent
                        child.set_value((self.predict(child.board)+ self.predict_two(child.board))/2)
                    else:
                        child.set_value(self.predict(child.board))
                else:   
                    child.add_value(self.evaluate(child.board))
                child.set_terminal(True)
                self.nodes[node].add_child(child.board.fen())
                self.nodes[child.board.fen()] = child
                if not board.is_game_over(claim_draw=True):
                    self.nodes[node].terminal = False
                    if self.heap_mark:
                        heapq.heappush(self.leaf_heapq, child)
                else:
                    self.nodes[node].terminal = True

        else:
            print("Error: node has no board")
            sys.exit(1)

    # returns - if good for the player that made move to get to this board, + if good for opponent
    def evaluate(self, board):
        # evaluate the board
        if isinstance(board, str):
            board = chess.Board(board)
        if board.is_game_over(claim_draw=True):
            if board.result() == "1-0" or board.result() == "0-1":
                return -1.3
            else:
                return 0
        elif board.is_checkmate():
            return -1.3
        else:
            return self.heuristic(board)
    
    # returns + if good for current player, - if good for opponent
    def heuristic(self, board):
        turn = 1
        if board.turn:
            turn = -1
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        if board.can_claim_threefold_repetition():
            return 1
        b_val = self.get_board_value(board)
        if abs(b_val) == 1:
            return b_val
        if self.value == None and self.heap_mark:
            #return max(-1, min((self.rollout(board) + b_val)/1.79, 1))
            return self.rollout(board)  + b_val
        elif self.value == None and not self.heap_mark:
            return self.rollout(board)
        return max(-1, min((self.nodes[board.fen()].value + b_val)/2, 1))


      
    def evaluate_material(self, board):
        material = 0
        try:
            material += len(board.pieces(chess.PAWN, chess.WHITE)) * 1
            material += len(board.pieces(chess.KNIGHT, chess.WHITE)) * 3
            material += len(board.pieces(chess.BISHOP, chess.WHITE)) * 3
            material += len(board.pieces(chess.ROOK, chess.WHITE)) * 5
            material += len(board.pieces(chess.QUEEN, chess.WHITE)) * 9
            material += len(board.pieces(chess.PAWN, chess.BLACK)) * -1
            material += len(board.pieces(chess.KNIGHT, chess.BLACK)) * -3
            material += len(board.pieces(chess.BISHOP, chess.BLACK)) * -3
            material += len(board.pieces(chess.ROOK, chess.BLACK)) * -5
            material += len(board.pieces(chess.QUEEN, chess.BLACK)) * -9
        except:
            return material
        return material
    
    def evaluate_position(self, board):
        position = 0
        try:
            # this is bad... should be evaluating the board position of the move
            # not the board position of the board
            for square in chess.SQUARES:
                position += len(board.attackers(chess.WHITE, square))*0.1
                position += len(board.attackers(chess.BLACK, square))*-0.1
                position += len(board.defenders(chess.WHITE, square))*0.1
                position += len(board.defenders(chess.BLACK, square))*-0.1

            # add bonus for center control
            position += len(board.attackers(chess.WHITE, chess.E4)) * 0.1
            position += len(board.attackers(chess.WHITE, chess.D4)) * 0.1
            position += len(board.attackers(chess.WHITE, chess.E5)) * 0.1
            position += len(board.attackers(chess.WHITE, chess.D5)) * 0.1
            position += len(board.attackers(chess.BLACK, chess.E4)) * -0.1
            position += len(board.attackers(chess.BLACK, chess.D4)) * -0.1
            position += len(board.attackers(chess.BLACK, chess.E5)) * -0.1
            position += len(board.attackers(chess.BLACK, chess.D5)) * -0.1
        except:
            return position
        return position
    
    def evaluate_mobility(self, board):
        mobility = 0
        try:
            mobility += len(board.legal_moves) * 0.1
        except:
            return mobility
        return mobility
    
    # + if good for player to play next move
    def rollout(self, board):
        # play out random moves until the game is over
        value = 0
        turn = -1
        if board.turn:
            turn = 1
        
        sims = self.num_simulations
        depth = self.max_depth

        """
        if self.move_count < 5:
            sims = int(self.num_simulations/15)
            depth = int(self.max_depth*15)
        elif self.move_count < 10:
            sims = int(self.num_simulations/10)
            depth = int(self.max_depth*10)
        elif self.move_count < 15:
            sims = int(self.num_simulations/5)
            depth = int(self.max_depth*5)
        """

        # sample up to depth
        if self.value != None and not self.expand_mode :
            for _ in range(sims):
                sim_board = chess.Board(board.fen())
                for i in range( random.randint(0, depth) ):
                    legal_moves = list(sim_board.legal_moves)
                    if len(legal_moves) == 0:
                        break
                    sim_board.push(random.choice(legal_moves))
                    # + if good for player to play next move
                b_value = self.get_board_value(sim_board)
                sim_turn = -1
                if sim_board.turn == board.turn:
                    sim_turn = 1
                if abs(b_value) == 1:
                    value += b_value*turn*sim_turn
                else:
                    # + if good for player to play next move
                    value -= self.predict(sim_board)*sim_turn*turn

            value = value/sims
        else:
            for _ in range(sims):
                sim_board = chess.Board(board.fen())
                for i in range( random.randint(0, depth)):
                    legal_moves = list(sim_board.legal_moves)
                    if len(legal_moves) == 0:
                        break
                    move = random.choice(legal_moves)
                    sim_board.push(move)
                sim_turn = -1
                if sim_board.turn == board.turn:
                    sim_turn = 1
                value -= self.get_board_value(sim_board)*turn*sim_turn
            value = value/sims

        return value
    
    # + if good for player to play next move
    def get_board_value(self, board):
        if board.result() == "1-0" or board.result == "0-1" or board.is_checkmate():
            return -1
        else:
            turn = -1
            if board.turn:
                turn = 1
            # probably better to return a value scaled by the depth and board_sum
            board_sum = self.evaluate_material(board) + self.evaluate_position(board)
            if -4 < board_sum < 0 :
                return -0.2*turn
            elif 4 > board_sum > 0:
                return 0.2*turn
            else:
                return board_sum*turn
            
    def backpropagate(self, node, value):
        # backpropagate the value of the board
        if node == None or self.nodes[node] == None:
            return
        self.nodes[node].add_visit(1)
        self.nodes[node].value = (self.nodes[node].value*self.nodes[node].visits - value)/(self.nodes[node].visits + 1)
        if self.nodes[node].parent != None:
            self.backpropagate(self.nodes[node].parent, -value)
        

    # returns - if good for parent, + if good for child (next to play)
    def predict(self, board):
        # use a neural network to predict the value of the board
        # for now, just return 0
        if type( board) == str:
            board = chess.Board(board)
        turn = -1
        if board.turn:
            turn = 1
        if self.value == None or self.model_input == None:
            return 0
        value = turn*(self.value.predict(np.array([ModelInput(self.model_input).get_input(board)]), verbose=0)[0][0]*2 - 1)
        return value
    
    def predict_two(self, board):
        # use a neural network to predict the value of the board
        # for now, just return 0
        if type( board) == str:
            board = chess.Board(board)
        turn = -1
        if board.turn:
            turn = 1
        if self.value_two == None or self.model_input == None:
            return 0
        value = turn*(self.value_two.predict(np.array([ModelInput(self.model_input).get_input(board)]), verbose=0)[0][0]*2 - 1)
        return value
    
    def best_move(self):
        # find the best move
        best_move = None
        best_value = math.inf
        checkmate = None
        for child in self.nodes[self.root].children:
            if self.nodes[child].value < best_value:
                best_move = self.nodes[child].action
                best_value = self.nodes[child].value
            if self.nodes[child].board.is_checkmate():
                checkmate = child.action
            if len(self.nodes[child].children) == 1:
                great_grandchild_checks = 0
                for great_grandchild in self.nodes[self.nodes[child].children[0]].children:
                    if self.nodes[great_grandchild].value < best_value:
                        best_move = self.nodes[child].action
                        best_value = self.nodes[great_grandchild].value
                    if self.nodes[great_grandchild].board.is_checkmate():
                        great_grandchild_checks += 1
                if great_grandchild_checks == len(self.nodes[self.nodes[child].children[0]].children):
                    checkmate = child.action
        self.best_move_value = best_value
        if checkmate != None:
            return checkmate
        return best_move
    
    def check_move(self, move):
        # check if the move is legal
        if move in self.nodes[self.root].board.legal_moves:
            return True
        # check if move is blunder
        # to-do : check if move is blunder

        # check if move is a mistake

        # check if move is a bad move

        return False
    
from functools import total_ordering
@total_ordering
class Node():
    def __init__(self):
        self.board = chess.Board()
        self.visits = 1
        self.value = 0
        self.children = []
        self.parent = None
        self.action = None
        self.player = None
        self.depth = 0
        self.terminal = True

    def set_board(self, board):
        self.board = chess.Board(board)
        # set self.player to the player who's turn it is
        if self.board.turn:
            self.player = -1
        else:
            self.player = 1
        
        #todo: set self.terminal to true if the game is over

    def add_visit(self, visits):
        self.visits += visits

    def set_value(self, value):
        self.value = value

    def add_value(self, value):
        self.value += value

    def add_child(self, child):
        if self.children == None:
            self.children = []
        self.children.append(child)


    def set_parent(self, parent):
        self.parent = parent

    def set_action(self, action):
        self.action = action

    def set_depth(self, depth):
        self.depth = depth

    def set_terminal(self, terminal):
        self.terminal = terminal

    def __repr__(self):
        return f"Node({self.board}, {self.visits}, {self.value}, {self.children}, {self.parent}, {self.action}, {self.player}, {self.depth}, {self.terminal})"
    
    def __str__(self):
        return f"Node({self.board}, {self.visits}, {self.value}, {self.children}, {self.parent}, {self.action}, {self.player}, {self.depth}, {self.terminal})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.value == other.value
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.value < other.value
        

def main():
    pass

if __name__ == "__main__":
    main()