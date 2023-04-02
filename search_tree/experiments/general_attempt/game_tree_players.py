# Chess Game Player

# This program is a chess game player that uses the minimax algorithm to
# determine the best move for the computer to make. The program uses the
# chess module to represent the chess board and pieces. The chess module
# can be found at https://pypi.org/project/python-chess/.


import chess
import chess.svg
import chess.polyglot
import chess.engine
import chess.pgn
import chess.syzygy
import pandas as pd

import random
import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import matplotlib.ticker as ticker

from IPython.display import SVG, display
from IPython.display import clear_output
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.display import display_pretty
from IPython.display import display_html
from IPython.display import Javascript
from IPython.display import Audio
from IPython.display import YouTubeVideo
from IPython.display import IFrame
from IPython.display import Latex
from IPython.display import Markdown
from IPython.display import Math
from IPython.display import Pretty
from IPython.display import clear_output
from IPython.display import set_matplotlib_formats
from IPython.display import set_matplotlib_close


class ChessGamePlayer:
    def __init__(self, board, depth, player_color, opponent_color):
        self.board = board
        self.depth = depth
        self.player_color = player_color
        self.opponent_color = opponent_color
        self.best_move = None
        self.best_move_value = None
        self.move_values = []
        self.move_values_dict = {}
        self.move_values_dict['move'] = []
        self.move_values_dict['value'] = []
        self.move_values_dict['depth'] = []
        self.move_values_dict['alpha'] = []
        self.move_values_dict['beta'] = []
        self.move_values_dict['alpha_beta'] = []
        self.move_values_dict['best_move'] = []
        self.move_values_dict['best_move_value'] = []
        self.move_values_dict['best_move_depth'] = []
        self.move_values_dict['best_move_alpha'] = []
        self.move_values_dict['best_move_beta'] = []
        self.move_values_dict['best_move_alpha_beta'] = []
        
    def get_best_move(self):
        return self.best_move
    
    def get_best_move_value(self):
        return self.best_move_value
    
    def get_move_values(self):
        return self.move_values
    
    def get_move_values_dict(self):
        return self.move_values_dict
    
    def get_move_values_dict_df(self):
        return pd.DataFrame(self.move_values_dict)
    
    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0:
            return self.evaluate_board(board)
        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def minimax_with_move(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0:
            return self.evaluate_board(board)
        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
        
    def minimax_with_move_and_values(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0:
            return self.evaluate_board(board)
        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move, alpha, beta
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move, alpha, beta


    def play(self):
        if self.player_color == chess.WHITE:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, self.depth, True, -math.inf, math.inf)
        else:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, self.depth, False, -math.inf, math.inf)
        self.board.push(self.best_move)
        return self.board
    
    def evaluate_board(self, board):
        if board.is_checkmate():
            if board.turn:
                return -math.inf
            else:
                return math.inf
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        if board.can_claim_threefold_repetition():
            return 0
        return self.evaluate_material(board) + self.evaluate_position(board)
    
    def evaluate_material(self, board):
        material = 0
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
        return material
    
    def evaluate_position(self, board):
        position = 0
        position += len(board.attackers(chess.WHITE, chess.E4)) * 0.1
        position += len(board.attackers(chess.WHITE, chess.D4)) * 0.1
        position += len(board.attackers(chess.WHITE, chess.E5)) * 0.1
        position += len(board.attackers(chess.WHITE, chess.D5)) * 0.1
        position += len(board.attackers(chess.BLACK, chess.E4)) * -0.1
        position += len(board.attackers(chess.BLACK, chess.D4)) * -0.1
        position += len(board.attackers(chess.BLACK, chess.E5)) * -0.1
        position += len(board.attackers(chess.BLACK, chess.D5)) * -0.1
        return position
    
    def search(self):
        if self.player_color == chess.WHITE:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, self.depth, True, -math.inf, math.inf)
        else:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, self.depth, False, -math.inf, math.inf)
        return self.best_move_value, self.best_move, alpha, beta
    
    def search_with_depth(self, depth):
        if self.player_color == chess.WHITE:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, depth, True, -math.inf, math.inf)
        else:
            self.best_move_value, self.best_move, alpha, beta = self.minimax_with_move_and_values(self.board, depth, False, -math.inf, math.inf)
        return self.best_move_value, self.best_move, alpha, beta
    

class HumanPlayer:
    def __init__(self, board, player_color):
        self.board = board
        self.player_color = player_color
    
    def play(self):
        while True:
            move = input()
            try:
                move = chess.Move.from_uci(move)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    break
                else:
                    print("Illegal move, try again")
            except ValueError:
                print("Invalid syntax, try again")
    
    def search(self):
        pass
    
    def search_with_depth(self, depth):
        pass


class ComputerPlayer:
    def __init__(self, board, player_color, depth):
        self.board = board
        self.player_color = player_color
        self.depth = depth
        self.best_move_value = 0
        self.best_move = chess.Move.null()
    
    def minimax(self, board, depth, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval)
            return min_eval
    
    def play(self):
        if self.player_color == chess.WHITE:
            self.best_move_value = self.minimax(self.board, self.depth, True)
        else:
            self.best_move_value = self.minimax(self.board, self.depth, False)
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.player_color == chess.WHITE:
                eval = self.minimax(self.board, self.depth - 1, False)
            else:
                eval = self.minimax(self.board, self.depth - 1, True)
            self.board.pop()
            if eval == self.best_move_value:
                self.best_move = move
                break
        self.board.push(self.best_move)
        return self.board
    
    def evaluate_board(self, board):
        if board.is_checkmate():
            if board.turn:
                return -math.inf
            else:
                return math.inf
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0
        if board.can_claim_threefold_repetition():
            return 0
        return self.evaluate_material(board) + self.evaluate_position(board)
    
    def evaluate_material(self, board):
        material = 0
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
        return material
    
    def evaluate_position(self, board):
        position = 0
        position += len(board.attackers(chess.WHITE, chess.E4)) * 0.1
        position += len(board.attackers(chess.WHITE, chess.D4)) * 0.1
        position += len(board.attackers(chess.BLACK, chess.E5)) * -0.1
        position += len(board.attackers(chess.BLACK, chess.D5)) * -0.1
        return position
    
    def search(self):
        if self.player_color == chess.WHITE:
            self.best_move_value = self.minimax(self.board, self.depth, True)
        else:
            self.best_move_value = self.minimax(self.board, self.depth, False)
        return self.best_move_value
    
    def search_with_depth(self, depth):
        if self.player_color == chess.WHITE:
            self.best_move_value = self.minimax(self.board, depth, True)
        else:
            self.best_move_value = self.minimax(self.board, depth, False)
        return self.best_move_value
    
    def search_with_move(self, move):
        self.board.push(move)
        if self.player_color == chess.WHITE:
            self.best_move_value = self.minimax(self.board, self.depth, False)
        else:
            self.best_move_value = self.minimax(self.board, self.depth, True)
        self.board.pop()
        return self.best_move_value
    


# main function

if __name__ == "__main__":
    board = chess.Board()
    player_color = chess.WHITE
    computer = ComputerPlayer(board, player_color, 4)
    computer_2 = ComputerPlayer(board, not player_color, 4)
    human = HumanPlayer(board, not player_color)
    while True:
        print(board)
        if board.turn == player_color:
            computer.play()
        else:
            computer_2.play()
        if board.is_game_over():
            break
    print(board)
    print(board.result())