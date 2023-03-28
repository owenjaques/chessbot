
import chess
import numpy as np
from feature_extraction.feature_v1 import get_features as get_features_v1

class SVMBot():
    def __init__(self, model):
        self.model = model
    
    def get_best_move(self, board):
        best_move_uci = None

        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                prediction = self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                prediction = self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        return best_move_uci
    
    def minimax(self, board, depth, maximizing_player):
        if depth == 0:
            return self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
        if maximizing_player:
            best_score = -np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.minimax(board, depth - 1, False)
                board.pop()
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.minimax(board, depth - 1, True)
                board.pop()
                best_score = min(score, best_score)
            return best_score
        
    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
        if maximizing_player:
            best_score = -np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                best_score = max(score, best_score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                best_score = min(score, best_score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_score
        
    def get_best_move_minimax(self, board, depth):
        best_move_uci = None
        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                prediction = self.minimax(board, depth, False)
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                prediction = self.minimax(board, depth, True)
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        return best_move_uci
    
    def get_best_move_alpha_beta(self, board, depth):
        best_move_uci = None
        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                prediction = self.alpha_beta(board, depth, -np.inf, np.inf, False)
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                prediction = self.alpha_beta(board, depth, -np.inf, np.inf, True)
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        return best_move_uci

        
    
    
