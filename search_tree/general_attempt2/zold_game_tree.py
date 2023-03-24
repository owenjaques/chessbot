# monte carlo tree search algorithm for a general game tree with a general game state
# uses a machine learning model for a specific game to evaluate the game state and predict the next move
# uses a loaded model to evaluate the game state and predict the next move for chess as an example
# a neural network modelt to predict where to rollout the game state
# uses the chess library to create a chess game state and check for legal moves
import chess

# uses the chess library to create a chess game tree
import chess.pgn

# uses the chess library to create a chess game state and check for legal moves
import chess.engine

import numpy as np
import random
import math
import time
import os
import sys
import pickle

# import the neural network model
from keras.models import load_model

from zold_chess_game_player import *

# class for a general game state evaluator and predictor for a specific game
class EvaluatorPredictor:
    # constructor
    def __init__(self, model):
        self.model = model

    # evaluate the game state
    def evaluate_game_state(self, game_state):
        return self.model.predict(game_state)

    # predict the next move
    def predict_next_move(self, game_state):
        return self.model.predict(game_state)
    
# class to create a game tree and search it using monte carlo tree search algorithm
class MCTS:
    # constructor
    def __init__(self, game_state, evaluator_predictor):
        self.game_state = game_state
        self.evaluator_predictor = evaluator_predictor
        self.game_tree = GameTreeNode(game_state)
        self.num_simulations = 0
        self.num_rollouts = 0
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0

    # get the number of simulations
    def get_num_simulations(self):
        return self.num_simulations
    
    # get the number of rollouts
    def get_num_rollouts(self):
        return self.num_rollouts
    
    # get the number of wins
    def get_num_wins(self):
        return self.num_wins
    
    # get the number of losses
    def get_num_losses(self):
        return self.num_losses
    
    # get the number of draws
    def get_num_draws(self):
        return self.num_draws
    
    # get the game tree
    def get_game_tree(self):
        return self.game_tree
    
    # get the game state
    def get_game_state(self):
        return self.game_state
    
    # get the evaluator predictor
    def get_evaluator_predictor(self):
        return self.evaluator_predictor
    
    # get the best move
    def get_best_move(self):
        return self.game_tree.get_child_with_best_win_rate_and_most_visits().get_move()
    
    # search the game tree using monte carlo tree search algorithm with expansion, rollout, and backpropagation
    def search(self, num_simulations):
        self.num_simulations = num_simulations
        for i in range(num_simulations):
            # expansion
            leaf_node = self.select(self.game_tree)
            # rollout
            winner = self.rollout(leaf_node)
            # backpropagation
            self.backpropagate(leaf_node, winner)

    # select a leaf node to expand
    def select(self, node):
        # if the node is a leaf node
        if node.get_num_children() == 0:
            return node
        # if the node is not a leaf node
        else:
            # if the node has unexplored children
            if len(node.get_game_state().get_legal_moves()) > node.get_num_children():
                return self.expand(node)
            # if the node has no unexplored children
            else:
                return self.select(node.get_child_with_best_win_rate_and_most_visits())
            
    # expand a node fully by creating all of its children up to a depth of 3
    def expand(self, node):
        # create all of the children
        for move in node.get_game_state().get_legal_moves():
            # create the child game state
            child_game_state = node.get_game_state().get_game_state().copy()
            child_game_state.push(move)
            # create the child game tree node
            child_node = GameTreeNode(child_game_state, move, node)
            # add the child to the node
            node.add_child(child_node)
        # return the last child
        return node.get_children()[-1]
    
    # rollout a game from a node
    def rollout(self, node):
        # create a copy of the game state
        game_state = node.get_game_state().get_game_state().copy()
        # while the game is not over
        while not game_state.is_game_over():
            # get the legal moves
            legal_moves = list(game_state.legal_moves)
            # pick a random move
            move = random.choice(legal_moves)
            # make the move
            game_state.push(move)
        # get the winner
        winner = game_state.result()
        # update the number of rollouts
        self.num_rollouts += 1
        # update the number of wins, losses, and draws
        if winner == '1-0':
            self.num_wins += 1
        elif winner == '0-1':
            self.num_losses += 1
        else:
            self.num_draws += 1
        # return the winner
        return winner
    
    # backpropagate the result of a rollout from a node
    def backpropagate(self, node, winner):
        # if the node is not the root node
        if node.get_parent() is not None:
            # update the number of visits
            node.increment_num_visits()
            # update the number of wins
            if winner == '1-0':
                node.increment_num_wins()
            # update the number of losses
            elif winner == '0-1':
                node.increment_num_losses()
            # update the number of draws
            else:
                node.increment_num_draws()
            # backpropagate the result of the rollout
            self.backpropagate(node.get_parent(), winner)

# class for a game tree node
class GameTreeNode:
    # constructor
    def __init__(self, game_state, move=None, parent=None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = []
        self.num_visits = 0
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0

    # get the game state
    def get_game_state(self):
        return self.game_state
    
    # get the move
    def get_move(self):
        return self.move
    
    # get the parent
    def get_parent(self):
        return self.parent
    
    # get the children
    def get_children(self):
        return self.children
    
    # get the number of children
    def get_num_children(self):
        return len(self.children)
    
    # get the number of visits
    def get_num_visits(self):
        return self.num_visits
    
    # get the number of wins
    def get_num_wins(self):
        return self.num_wins
    
    # get the number of losses
    def get_num_losses(self):
        return self.num_losses
    
    # get the number of draws
    def get_num_draws(self):
        return self.num_draws
    
    # get the win rate
    def get_win_rate(self):
        if self.num_wins + self.num_losses + self.num_draws == 0:
            return 0
        else:
            return (self.num_wins + 0.5 * self.num_draws) / (self.num_wins + self.num_losses + self.num_draws)
    
    # get the child with the best win rate and most visits
    def get_child_with_best_win_rate_and_most_visits(self):
        best_child = self.children[0]
        for child in self.children:
            if child.get_win_rate() > best_child.get_win_rate():
                best_child = child
            elif child.get_win_rate() == best_child.get_win_rate():
                if child.get_num_visits() > best_child.get_num_visits():
                    best_child = child
        return best_child
    
    # add a child
    def add_child(self, child):
        self.children.append(child)
    
    # increment the number of visits
    def increment_num_visits(self):
        self.num_visits += 1
    
    # increment the number of wins
    def increment_num_wins(self):
        self.num_wins += 1

    # increment the number of losses
    def increment_num_losses(self):
        self.num_losses += 1

    # increment the number of draws
    def increment_num_draws(self):
        self.num_draws += 1

# class for a game state
class GameState:
    # constructor
    def __init__(self, game_state):
        self.game_state = game_state

    # get the game state
    def get_game_state(self):
        return self.game_state

    # get the legal moves
    def get_legal_moves(self):
        return list(self.game_state.legal_moves)
    
    # get the result
    def get_result(self):
        return self.game_state.result()
    
    # get the winner
    def get_winner(self):
        return self.game_state.result()
    
    # get the board
    def get_board(self):
        return self.game_state.board
    
    # get the number of pieces
    def get_num_pieces(self):
        return len(self.game_state.board.pieces)
    
    # get the number of pieces for a color
    def get_num_pieces_for_color(self, color):
        return len(self.game_state.board.pieces[color])
    
    # get the number of pieces for a piece type
    def get_num_pieces_for_piece_type(self, piece_type):
        return len(self.game_state.board.pieces[piece_type])
    
    # get the number of pieces for a color and piece type
    def get_num_pieces_for_color_and_piece_type(self, color, piece_type):
        return len(self.game_state.board.pieces[color][piece_type])
    

# class for a game state evaluator
class GameStateEvaluator:
    # constructor
    def __init__(self):
        pass
    
    # evaluate the game state
    def evaluate_game_state(self, game_state):
        # get the winner
        winner = game_state.get_winner()
        # if the winner is white
        if winner == '1-0':
            return 1
        # if the winner is black
        elif winner == '0-1':
            return -1
        # if the winner is a draw
        else:
            return 0
        

# class for a chess game state generator
class GameStateGenerator:
    # constructor
    def __init__(self):
        pass
    
    # generate a new game state
    def generate_new_game_state(self, game_state, move):
        # get the board
        board = game_state.get_board()
        # get the piece type
        piece_type = move.piece
        # get the color
        color = move.color
        # get the from square
        from_square = move.from_square
        # get the to square
        to_square = move.to_square
        # get the promotion
        promotion = move.promotion
        # get the capture square
        capture_square = move.capture_square
        # get the is en passant
        is_en_passant = move.is_en_passant
        # get the is castling
        is_castling = move.is_castling
        # get the is promotion
        is_promotion = move.is_promotion
        # get the is check
        is_check = move.is_check
        # get the is checkmate
        is_checkmate = move.is_checkmate
        # get the is stalemate
        is_stalemate = move.is_stalemate
        # get the is insufficient material
        is_insufficient_material = move.is_insufficient_material
        # get the is fifty move rule
        is_fifty_move_rule = move.is_fifty_move_rule
        # get the is threefold repetition
        is_threefold_repetition = move.is_threefold_repetition
        # get the is fifty move rule
        is_fifty_move_rule = move.is_fifty_move_rule

        # create a new move
        new_move = chess.Move(from_square, to_square, promotion=promotion)
        # create a new board
        new_board = board.copy()
        # make the move
        new_board.push(new_move)
        # create a new game state
        new_game_state = chess.Board(new_board.fen())
        # return the new game state
        return new_game_state
    
        
# class to represent a chess game
class ChessGame:
    # constructor
    def __init__(self, white_player, black_player):
        self.white_player = white_player
        self.black_player = black_player
        self.game_state = chess.Board()
        self.game_state_evaluator = GameStateEvaluator()
        self.game_state_generator = GameStateGenerator()
        self.game_over = False
        
    # play the game
    def play_game(self):
        # while the game is not over
        while not self.game_over:
            # print the game state
            print(self.game_state)
            # if it is white's turn
            if self.game_state.turn:
                # get the best move
                best_move = self.white_player.get_best_move(self.game_state)
                # make the move
                self.game_state.push(best_move)
            # else it is black's turn
            else:
                # get the best move
                best_move = self.black_player.get_best_move(self.game_state)
                # make the move
                self.game_state.push(best_move)
            # if the game is over
            if self.game_state.is_game_over():
                # set the game over flag
                self.game_over = True
                # print the game state
                print(self.game_state)
                # print the result
                print(self.game_state.result())

# class to represent a chess game player that uses minimax


class MinimaxChessGamePlayer(ChessGamePlayer):
    # constructor
    def __init__(self, game_state_evaluator, game_state_generator, depth):
        self.game_state_evaluator = game_state_evaluator
        self.game_state_generator = game_state_generator
        self.depth = depth

    # get the best move
    def get_best_move(self, game_state):
        # get the legal moves
        legal_moves = game_state.get_legal_moves()
        # get the best move
        best_move = legal_moves[0]
        # get the best move score
        best_move_score = -1
        # for each legal move
        for legal_move in legal_moves:
            # get the new game states for the legal move
            new_game_state = self.game_state_generator.generate_new_game_state(
                game_state, legal_move)
            # get the new game state score
            new_game_state_score = self.minimax(
                new_game_state, self.depth, False)
            # if the new game state score is better than the best move score
            if new_game_state_score > best_move_score:
                # set the best move score
                best_move_score = new_game_state_score
                # set the best move
                best_move = legal_move
        # return the best move
        return best_move

    # minimax
    def minimax(self, game_state, depth, is_maximizing):
        # if the depth is 0 or the game is over
        if depth == 0 or game_state.get_result() != None:
            # return the game state score
            return self.game_state_evaluator.evaluate_game_state(game_state)
        # if the player is maximizing
        if is_maximizing:
            # get the legal moves
            legal_moves = game_state.get_legal_moves()
            # get the best move score
            best_move_score = -1
            # for each legal move
            for legal_move in legal_moves:
                # get the new game state
                new_game_state = self.game_state_generator.generate_new_game_state(
                    game_state, legal_move)
                # get the new game state score
                new_game_state_score = self.minimax(
                    new_game_state, depth - 1, False)
                # if the new game state score is better than the best move score
                if new_game_state_score > best_move_score:
                    # set the best move score
                    best_move_score = new_game_state_score
            # return the best move score
            return best_move_score
        # else the player is minimizing
        else:
            # get the legal moves
            legal_moves = game_state.get_legal_moves()
            # get the best move score
            best_move_score = 1
            # for each legal move
            for legal_move in legal_moves:
                # get the new game state
                new_game_state = self.game_state_generator.generate_new_game_state(
                    game_state, legal_move)
                # get the new game state score
                new_game_state_score = self.minimax(
                    new_game_state, depth - 1, True)
                # if the new game state score is better than the best move score
                if new_game_state_score < best_move_score:
                    # set the best move score
                    best_move_score = new_game_state_score
            # return the best move score
            return best_move_score


# class to represent a chess game player that uses alpha-beta pruning
class AlphaBetaPruningChessGamePlayer(ChessGamePlayer):
    # constructor
    def __init__(self, game_state_evaluator, game_state_generator, depth):
        self.game_state_evaluator = game_state_evaluator
        self.game_state_generator = game_state_generator
        self.depth = depth

    # get the best move
    def get_best_move(self, game_state):
        # get the legal moves
        legal_moves = game_state.get_legal_moves()
        # get the best move
        best_move = legal_moves[0]
        # get the best move score
        best_move_score = -1
        # for each legal move
        for legal_move in legal_moves:
            # get the new game state
            new_game_state = self.game_state_generator.generate_new_game_state(
                game_state, legal_move)
            # get the new game state score
            new_game_state_score = self.alpha_beta_pruning(
                new_game_state, self.depth, -1, 1, False)
            # if the new game state score is better than the best move score
            if new_game_state_score > best_move_score:
                # set the best move score
                best_move_score = new_game_state_score
                # set the best move
                best_move = legal_move
        # return the best move
        return best_move

    # alpha-beta pruning
    def alpha_beta_pruning(self, game_state, depth, alpha, beta, is_maximizing):
        # if the depth is 0 or the game is over
        if depth == 0 or game_state.get_result() != None:
            # return the game state score
            return self.game_state_evaluator.evaluate_game_state(game_state)
        # if the player is maximizing
        if is_maximizing:
            # get the legal moves
            legal_moves = game_state.get_legal_moves()
            # get the best move score
            best_move_score = -1
            # for each legal move
            for legal_move in legal_moves:
                # get the new game state
                new_game_state = self.game_state_generator.generate_new_game_state(
                    game_state, legal_move)
                # get the new game state score
                new_game_state_score = self.alpha_beta_pruning(
                    new_game_state, depth - 1, alpha, beta, False)
                # if the new game state score is better than the best move score
                if new_game_state_score > best_move_score:
                    # set the best move score
                    best_move_score = new_game_state_score
                # if the best move score is greater than or equal to the beta
                if best_move_score >= beta:
                    # return the best move score
                    return best_move_score
                # if the best move score is greater than the alpha
                if best_move_score > alpha:
                    # set the alpha
                    alpha = best_move_score
            # return the best move score
            return best_move_score
        # else the player is minimizing
        else:
            # get the legal moves
            legal_moves = game_state.get_legal_moves()
            # get the best move score
            best_move_score = 1
            # for each legal move
            for legal_move in legal_moves:
                # get the new game state
                new_game_state = self.game_state_generator.generate_new_game_state(
                    game_state, legal_move)
                # get the new game state score
                new_game_state_score = self.alpha_beta_pruning(
                    new_game_state, depth - 1, alpha, beta, True)
                # if the new game state score is better than the best move score
                if new_game_state_score < best_move_score:
                    # set the best move score
                    best_move_score = new_game_state_score
                # if the best move score is less than or equal to the alpha
                if best_move_score <= alpha:
                    # return the best move score
                    return best_move_score
                # if the best move score is less than the beta
                if best_move_score < beta:
                    # set the beta
                    beta = best_move_score
            # return the best move score
            return best_move_score

# class to represent a random chess game player


class RandomChessGamePlayer(ChessGamePlayer):
    # constructor
    def __init__(self):
        pass

    # get the best move
    def get_best_move(self, game_state):
        # get the legal moves
        legal_moves = list(game_state.legal_moves)
        # return a random legal move
        return random.choice(legal_moves)


# main
if __name__ == '__main__':
    # create a chess game
    chess_game = ChessGame(RandomChessGamePlayer(), RandomChessGamePlayer())
    # play the game
    chess_game.play_game()
    

    
