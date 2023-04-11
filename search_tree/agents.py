# chess agent classes and functions for the chess tournament
# Path: search_tree\agents.py

import chess
import random
import numpy as np
import os
import sys
import time
import datetime
import argparse
import chess_tournament

import keras

from MCTS import MCTS
from MCTSTest import MCTSTest
from neural_networks.chessbot.chessbot import ChessBot
from neural_networks.chessbot.modelinput import ModelInput

import chess
import os
#import stockfish
from stockfish import Stockfish

import sys 
sys.path.append('..')


#####################
####Black/White test

class White:
    def __init__(self):
        self.name = "White"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('simple'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('simple'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
class Black:
    def __init__(self):
        self.name = "Black"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('simple'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('simple'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)



class MCTSOwenBtfSimpleTEST:
    def __init__(self):
        self.name = "MCTSOwenBtfSimpleTEST!!!"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.model_two = keras.models.load_model('bin/owen/simple_input_model/model')
        self.searcher = MCTSTest(max_time = 30, use_heap=True)
        self.color = chess.BLACK
    def initialize(self, color):
        self.color = color
        pass
    def get_move(self, board):
        self.searcher = MCTSTest(max_time = 30, color=self.color, value_nn=self.model, value_nn_2=self.model_two, model_input='simple', use_heap=False)
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

class MCTSAgentTEST:
    def __init__(self):
        self.name = "MCTSAgenttime30TEST!!!"
        self.searcher = MCTSTest(max_time = 30, use_heap=False)
    def initialize(self, color):
        
        pass
    def get_move(self, board):
        # get the best move
        self.searcher = MCTSTest(max_time = 30, use_heap=True)
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)


###########################################################################################
##################################  The agent to beat  ####################################

class StockfishAgent1350:
    def __init__(self):
        self.path = os.getcwd()+"\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe"
        self.stockfish = Stockfish(self.path, parameters={"UCI_LimitStrength": True})
        self.name = "Stockfish1350"
    def initialize(self, color):
        pass
    def get_move(self, board):
        self.stockfish.set_fen_position(board.fen())
        self.stockfish.update_engine_parameters({"UCI_Elo:": 1350})
        move = self.stockfish.get_best_move()
        return move
    
class StockfishAgent1500:
    def __init__(self):
        self.path = os.getcwd()+"/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe"
        self.stockfish = Stockfish(self.path, parameters={"Threads": 1, "UCI_LimitStrength": True, "Skill Level": 20})
        self.name = "Stockfish1500"
    def initialize(self, color):
        pass
    def get_move(self, board):
        self.stockfish.set_fen_position(board.fen())
        self.stockfish.update_engine_parameters({"UCI_Elo:": 1500})
        move = self.stockfish.get_best_move()
        return move

class StockfishAgent1650:
    def __init__(self):
        self.path = os.getcwd()+"/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe"
        self.stockfish = Stockfish(self.path, parameters={"Threads": 1, "UCI_LimitStrength": True, "Skill Level": 20})
        self.name = "Stockfish1650"
    def initialize(self, color):
        pass
    def get_move(self, board):
        self.stockfish.set_fen_position(board.fen())
        self.stockfish.update_engine_parameters({"UCI_Elo:": 1650})
        move = self.stockfish.get_best_move()
        return move
    




###########################################################################################
#################################### MCTS AGENTS ##########################################

# MCTS agent on its own with no neural network
# uses heap method for tree search
# settings: default settings 
class MCTSHeapAgent:
    def __init__(self):
        self.name = "MCTSHeapAgenttime30"
        self.searcher = MCTS(max_time = 30, use_heap=False)
    def initialize(self, color):
        self.searcher = MCTS(max_time = 30, use_heap=True)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

# MCTS agent on its own with no neural network
# uses heap method for tree search
# settings: num_simulations = 2000, max_depth = 15
class MCTSHeapAgent2000and15:
    def __init__(self):
        self.name = "MCTSHeapAgent_time60_num_simulations2000_max_depth15"
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self, color):
        self.searcher = MCTS(max_time = 60, num_simulations=2000, max_depth=15, use_heap=True)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move    
    def get_next_move(self, board):
        return self.get_move(board)
    
# MCTS agent on its own with no neural network
# uses UCT selection method for tree search
# settings: num_simulations = 2000, max_depth = 15
class MCTSAgent:
    def __init__(self):
        self.name = "MCTSAgent_time30_num_simulations1000_max_depth25"
        self.searcher = MCTS(max_time = 30, use_heap=True)
    def initialize(self,color):
        pass
    def get_move(self, board):
        # get the best move
        self.searcher = MCTS(max_time = 30,  use_heap=False)
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
    
###########################################################################################
################################# MCTS w/ NN AGENTS #######################################

# MCTS with Owens Simple Input Neural Network

class MCTSBtfSingle:
    def __init__(self):
        self.name = "MCTSBTFSingleInput"
        self.model = keras.models.load_model('bin/btf/single_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self,color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, model_input='positions', use_heap=False)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)


class MCTSBtfSimple:
    def __init__(self):
        self.name = "MCTSBTFSimpleInput"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self,color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, model_input='simple', use_heap=False)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
class MCTSOwenSingle:
    def __init__(self):
        self.name = "MCTSOwenSingleInput"
        self.model = keras.models.load_model('bin/owen/single_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self,color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, model_input='positions', use_heap=False)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

    
class MCTSOwenSimple:
    def __init__(self):
        self.name = "MCTSOwenSimpleInput"
        self.model = keras.models.load_model('bin/owen/simple_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self,color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, model_input='simple', use_heap=False)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

class MCTSOwenBtfSimple:
    def __init__(self):
        self.name = "MCTSOwenBtfSimpleInput"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.model_two = keras.models.load_model('bin/owen/simple_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self, color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, value_nn_2=self.model_two, model_input='simple', use_heap=False)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
class MCTSOwenBtfSimpleExpand:
    def __init__(self):
        self.name = "MCTSOwenBtfSimpleInputExpandmode"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.model_two = keras.models.load_model('bin/owen/simple_input_model/model')
        self.searcher = MCTS(max_time = 60, use_heap=True)
    def initialize(self, color):
        self.searcher = MCTS(max_time = 60, value_nn=self.model, value_nn_2=self.model_two, model_input='simple', use_heap=False, expand_mode=True)
        pass
    def get_move(self, board):
        # get the best move
        move = self.searcher.search(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)



###########################################################################################
###################################### NN AGENTS ##########################################

class ChessBotAgentBtfSingleInput:
    def __init__(self):
        self.name = "BTFSingleAgent"
        self.model = keras.models.load_model('bin/btf/single_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('positions'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('positions'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

class ChessBotAgentBtfSimpleInput:
    def __init__(self):
        self.name = "BTFSimpleAgent"
        self.model = keras.models.load_model('bin/btf/simple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('simple'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('simple'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
class ChessBotAgentBtfTripleInput:
    def __init__(self):
        self.name = "BTFTripleAgent"
        self.model = keras.models.load_model('bin/btf/triple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('all'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('all'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    

class ChessBotAgentOwenSingleInput:
    def __init__(self):
        self.name = "OWENSingleAgent"
        self.model = keras.models.load_model('bin/owen/single_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('positions'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('positions'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)

class ChessBotAgentOwenSimpleInput:
    def __init__(self):
        self.name = "OWENSimpleAgent"
        self.model = keras.models.load_model('bin/owen/simple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('simple'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('simple'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)
    
class ChessBotAgentOwenTripleInput:
    def __init__(self):
        self.name = "OWENTripleAgent"
        self.model = keras.models.load_model('bin/owen/triple_input_model/model')
        self.bot = ChessBot(self.model, ModelInput('all'), chess.WHITE, exploration_rate=0.0)
    def initialize(self, color):
        self.bot = ChessBot(self.model, ModelInput('all'), color, exploration_rate=0.0)
    def get_move(self, board):
        # get the best move
        move = self.bot.move(board)
        # return the best move
        return move
    def get_next_move(self, board):
        return self.get_move(board)


###########################################################################################
#################################### OTHER AGENTS #########################################
# (copilot) : 

# various agents for ours to play against : currently untested...
# each chess agent is initialized with a name
# the name is used to identify the agent in the tournament results
# each agent needs to have a get_move function that takes a chess board as input and returns a chess move
# random agent
class RandomAgent:
    def __init__(self):
        self.name = "RandomAgent"
    def initialize(self, color):
        pass
    def get_move(self, board):
        return random.choice(list(board.legal_moves))
    def get_next_move(self, board):
        return self.get_move(board)
    
# minimax agent : copilot directed from ??? github.com/kevinhughes27/TensorKart/blob/master/tensorkart/agents.py
class MinimaxAgent:
    def __init__(self):
        self.name = "MinimaxAgentDepth10"
        self.depth = 10
    def initialize(self, color):
        pass
    def get_move(self, board):
        return minimax(board, self.depth)[1]
    def get_next_move(self, board):
        return self.get_move(board)
    
# minimax agent with alpha beta pruning
class MinimaxABAgent:
    def __init__(self):
        self.name = "MinimaxABAgentDepth10"
        self.depth = 10
    def initialize(self, color):
        pass
    def get_move(self, board):
        return minimax(board, self.depth, True)[1]
    def get_next_move(self, board):
        return self.get_move(board)
    

# minimax function
# returns the best move and the best score for a given board and depth
# if alpha_beta is true, alpha beta pruning is used
def minimax(board, depth, alpha_beta = False):
    # if the depth is 0 or the game is over, return the score of the board
    if depth == 0 or board.is_game_over():
        return (score_board(board), None)
    # get the legal moves
    moves = list(board.legal_moves)
    best_move = random.choice(moves)
    # if it is white's turn
    if board.turn:
        # set the best score to the lowest possible score
        best_score = -9999
        # loop through the moves
        for move in moves:
            # make the move
            board.push(move)
            # get the score of the board
            score = minimax(board, depth - 1, alpha_beta)[0]
            # if alpha beta pruning is enabled
            if alpha_beta:
                # if the score is greater than the best score
                if score > best_score:
                    # set the best score to the score
                    best_score = score
                    # set the best move to the move
                    best_move = move
                # if the score is less than the best score
                elif score < best_score:
                    # undo the move
                    board.pop()
                    # return the best score and best move
                    return (best_score, best_move)
            # if alpha beta pruning is not enabled
            else:
                # if the score is greater than the best score
                if score > best_score:
                    # set the best score to the score
                    best_score = score
                    # set the best move to the move
                    best_move = move
            # undo the move
            board.pop()
        # return the best score and best move
        return (best_score, best_move)
    # if it is black's turn
    else:
        # set the best score to the highest possible score
        best_score = 9999
        # loop through the moves
        for move in moves:
            # make the move
            board.push(move)
            # get the score of the board
            score = minimax(board, depth - 1, alpha_beta)[0]
            # if alpha beta pruning is enabled
            if alpha_beta:
                # if the score is less than the best score
                if score < best_score:
                    # set the best score to the score
                    best_score = score
                    # set the best move to the move
                    best_move = move
                # if the score is greater than the best score
                elif score > best_score:
                    # undo the move
                    board.pop()
                    # return the best score and best move
                    return (best_score, best_move)
            # if alpha beta pruning is not enabled
            else:
                # if the score is less than the best score
                if score < best_score:
                    # set the best score to the score
                    best_score = score
                    # set the best move to the move
                    best_move = move
            # undo the move
            board.pop()
        # return the best score and best move
        return (best_score, best_move)

    
# score the board
# the score is the difference between the number of white pieces and the number of black pieces
def score_board(board):
    material = 0
    try:
        material += len(board.pieces(chess.PAWN, chess.WHITE)) * 1
        material += len(board.pieces(chess.KNIGHT, chess.WHITE)) * 4
        material += len(board.pieces(chess.BISHOP, chess.WHITE)) * 3
        material += len(board.pieces(chess.ROOK, chess.WHITE)) * 5
        material += len(board.pieces(chess.QUEEN, chess.WHITE)) * 9
        material += len(board.pieces(chess.PAWN, chess.BLACK)) * -1
        material += len(board.pieces(chess.KNIGHT, chess.BLACK)) * -4
        material += len(board.pieces(chess.BISHOP, chess.BLACK)) * -3
        material += len(board.pieces(chess.ROOK, chess.BLACK)) * -5
        material += len(board.pieces(chess.QUEEN, chess.BLACK)) * -9
    except:
        return material
    return material

