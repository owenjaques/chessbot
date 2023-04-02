# MCTS for chess game using python-chess library
# shouldn't copy code from Github
#
# should use a neural network to evaluate the board

# https://joshvarty.github.io/AlphaZero/
# https://stackoverflow.com/questions/54027861/using-queue-priorityqueue-not-caring-about-comparisons

import chess
import random
import math
import time
import numpy as np
import copy
import sys
import os
import pickle
import queue
import heapq

from collections import defaultdict


class MCTS():
    def __init__(self, board, time_limit,num_simulations, max_depth=100, policy_nn=None, value_nn=None):
        self.board = board
        self.time_limit = time_limit
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.policy = policy_nn
        self.value = value_nn
        self.nodes = {}
        self.root = self.board.fen()
        self.heap = []
        self.nodes[self.root] = Node()
        self.nodes[self.root].set_board(self.root)
        self.nodes[self.root].set_visits(0)
        self.nodes[self.root].set_value(0)
        self.nodes[self.root].set_parent(None)
        self.nodes[self.root].set_action(None)
        self.nodes[self.root].set_depth(0)
        self.leaf_heapq = []
        heapq.heapify(self.leaf_heapq)
        self.nodes[self.root].children = self.expand( self.board.fen() )
        
    

    def search(self):
        start_time = time.time()
        # setup workers that will run simulations in parallel for a certain amount of time
        # workers will work on a queue of nodes to simulate and update the dictionary of nodes

        self.expand(self.root)

        # while the time limit has not been reached
        while time.time() - start_time < self.time_limit:
            # get the next node to simulate
            node = heapq.heappop(self.leaf_heapq).board.fen()

            if self.nodes[node].terminal:
                continue

            # expand the node
            self.expand(node)

            # backpropagate the value
            self.backpropagate(node)

        # return the best move
        return self.best_move()


    def expand(self, node):
        # expand the node and add the children to the heap
        board_start = chess.Board(self.nodes[node].board.fen())
        if self.nodes[node].terminal:
            return
        if board_start != None:
            # to do: add a policy network to evaluate the board
            legal_moves = list(board_start.legal_moves)

            for move in legal_moves:
                child = Node()
                board = chess.Board(board_start.fen())
                board.push(move)
                child.set_board(board.fen())
                child.set_parent(self.nodes[node].board.fen())
                child.set_action(move)
                child.set_depth(self.nodes[node].depth + 1)
                child.set_value(self.evaluate(child.board))
                self.nodes[node].add_child(child.board.fen())
                self.nodes[child.board.fen()] = child      
                heapq.heappush(self.leaf_heapq, child)

        else:
            print("Error: node has no board")
            sys.exit(1)

    def evaluate(self, board):
        # evaluate the board
        if board.is_game_over():
            if board.result() == "1-0":
                return 1
            elif board.result() == "0-1":
                return -1
            else:
                return 0
        else:
            return self.heuristic(board)
    
    def heuristic(self, board):
        if board.is_checkmate():
            if board.turn:
                return -1
            else:
                return 1
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        if board.can_claim_threefold_repetition():
            return 0
        if self.value == None:
            return self.rollout(board)
        # not sure if this is the best way to do this
        # try with just value and evaluate_material, evaluate_position
        return self.predict(board) 
    
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

    def rollout(self, board):
        # play out random moves until the game is over
        value = 0
        for _ in range(self.num_simulations):
            sim_board = chess.Board(board.fen())
            for i in range(200):
                legal_moves = list(sim_board.legal_moves)
                if len(legal_moves) == 0:
                    break
                move = random.choice(legal_moves)
                sim_board.push(move)
            value += self.get_board_value(sim_board)
        return value

    def get_board_value(self, board):
        if board.result() == "1-0":
            return 1
        elif board.result() == "0-1":
            return -1
        elif board.is_checkmate():
            if board.turn:
                return 1
            else:
                return -1
        else:
            board_sum = self.evaluate_material(board) + self.evaluate_position(board)
            if board_sum > 0:
                return 0.3
            elif board_sum < 0:
                return -0.3
            else:
                return 0
        
    def backpropagate(self, node):
        if self.nodes[node].parent == None:
            return
        self.nodes[node].visits += 1
        self.nodes[node].value += self.nodes[node].value / self.nodes[node].visits
        self.backpropagate(self.nodes[node].parent)

    def predict(self, board):
        # use a neural network to predict the value of the board
        # for now, just return 0
        return 0
    
    def best_move(self):
        # find the best move
        best_move = None
        best_value = math.inf
        for child in self.nodes[self.root].children:
            if self.nodes[child].value < best_value:
                best_move = self.nodes[child].action
                best_value = self.nodes[child].value

        return best_move
    

from functools import total_ordering

@total_ordering
class Node():
    def __init__(self):
        self.board = chess.Board()
        self.visits = 0
        self.value = 0
        self.children = []
        self.parent = None
        self.action = None
        self.player = None
        self.depth = 0
        self.terminal = False

    def set_board(self, board):
        self.board = chess.Board(board)
        # set self.player to the player who's turn it is
        if self.board.turn:
            self.player = -1
        else:
            self.player = 1
        
        #todo: set self.terminal to true if the game is over

    def set_visits(self, visits):
        self.visits = visits

    def set_value(self, value):
        self.value = value

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
        



# test codde

def test():
    board = chess.Board()
    mcts = MCTS(board, time_limit=10, num_simulations=10)
    mcts.search()
    print(mcts.best_move())


# main 

if __name__ == "__main__":
    test()