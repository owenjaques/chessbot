# MCTS for chess game using python-chess library
# shouldn't copy code from Github
#
# should use a neural network to evaluate the board

# https://joshvarty.github.io/AlphaZero/
# https://stackoverflow.com/questions/54027861/using-queue-priorityqueue-not-caring-about-comparisons


# to-do: heapq is not a great idea here...


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
    def __init__(self, time_limit, num_simulations, player='white', max_depth=100, policy_nn=None, value_nn=None, heap_mark=False):
        self.board = chess.Board()
        self.player = player
        self.time_limit = time_limit
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.policy = policy_nn
        self.value = value_nn
        self.heap_mark = heap_mark
        self.nodes = {}

    def set_root(self, board):
        self.root = board.fen()
        self.nodes[self.root] = Node()
        self.nodes[self.root].set_board(self.root)
        self.nodes[self.root].set_visits(1)
        self.nodes[self.root].set_value(0)
        self.nodes[self.root].set_parent(None)
        self.nodes[self.root].set_action(None)
        self.nodes[self.root].set_depth(0)
        if self.heap_mark:
            self.leaf_heapq = [self.nodes[self.root]]
            heapq.heapify(self.leaf_heapq)
        if self.root not in self.nodes:
            self.expand( self.board.fen() )
        
    def search(self, board):
        start_time = time.time()

        # set the root node
        self.set_root(board)

        node = self.nodes[self.root]

        # while the time limit has not been reached
        while time.time() - start_time < self.time_limit:
            # get the next node to simulate
            if self.heap_mark:
                node = heapq.heappop(self.leaf_heapq).board.fen()
            else:
                node = self.root
                while not self.nodes[node].terminal:
                    node = self.select(node)
                    if node == None:
                        break

            if node != None:
                # expand the node
                self.expand(node)

                value = sum(self.nodes[child].value for child in self.nodes[node].children)

                # backpropagate the value
                self.backpropagate(node, value)

        # return the best move
        return self.best_move()
    

    def select(self, node):
        # select the best child of the node
        best_child = None
        best_value = math.inf

        for child in self.nodes[node].children:
            try:
                child_value = self.nodes[child].value / self.nodes[child].visits + math.sqrt(2 * math.log(self.nodes[node].visits) / self.nodes[child].visits)
            except:
                child_value = self.evaluate(chess.Board(self.nodes[child].board))
                continue
            if child_value < best_value:
                best_value = child_value
                best_child = child

        return best_child
    
    def expand(self, node):
        # expand the node and add the children to the heap
        board_start = chess.Board(self.nodes[node].board.fen())

        if board_start != None:
            # to do: add a policy network to evaluate the board
            legal_moves = list(board_start.legal_moves)

            # currently.... is supposed to only expand one but I like the idea of expanding all
            # doesnt work well though. Is a stop gap measure until we have a policy network
            for move in legal_moves:
                child = Node()
                board = chess.Board(board_start.fen())
                board.push(move)
                child.set_board(board.fen())
                child.set_parent(self.nodes[node].board.fen())
                child.set_action(move)
                child.set_depth(self.nodes[node].depth + 1)
                child.add_value(self.evaluate(child.board))
                self.nodes[node].add_child(child.board.fen())
                self.nodes[child.board.fen()] = child   
                self.nodes[node].terminal = False
                if self.heap_mark:
                    heapq.heappush(self.leaf_heapq, child)

        else:
            print("Error: node has no board")
            sys.exit(1)

    def evaluate(self, board):
        # evaluate the board
        if board.is_game_over():
            if board.result() == "1-0" or board.result() == "0-1":
                return -1
            else:
                return 0
        else:
            return self.heuristic(board)
    
    def heuristic(self, board):
        if board.is_checkmate():
            return -1
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        if board.can_claim_threefold_repetition():
            return 0
        if self.value == 0:
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
        turn = -1
        if board.turn:
            turn = 1
        for _ in range(self.num_simulations):
            sim_board = chess.Board(board.fen())
            for i in range(self.max_depth):
                legal_moves = list(sim_board.legal_moves)
                if len(legal_moves) == 0:
                    break
                move = random.choice(legal_moves)
                sim_board.push(move)
            value += self.get_board_value(sim_board)*turn
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
        
    def backpropagate(self, node, value):
        # backpropagate the value of the board
        self.nodes[node].visits += 1
        self.nodes[node].value += value / self.nodes[node].visits+1
        if self.nodes[node].parent != None:
            self.backpropagate(self.nodes[node].parent, value)

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

    def set_visits(self, visits):
        self.visits = visits

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