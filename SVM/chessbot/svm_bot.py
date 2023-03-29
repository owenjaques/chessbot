
import chess
import numpy as np
from feature_extraction.feature_v1 import get_features as get_features_v1
from feature_extraction.feature_v2 import get_features as get_features_v2
from feature_extraction.feature_v3 import get_features as get_features_v3

class SVMBot():
    def __init__(self, model):
        self.model = model
    
    def get_best_move(self, board, feature_version=1):
        best_move_uci = None

        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                if feature_version == 1:
                    prediction = self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
                elif feature_version == 2:
                    prediction = self.model.predict(np.array(get_features_v2(board)).reshape(1, -1))
                elif feature_version == 3:
                    prediction = self.model.predict(np.array(get_features_v3(board)).reshape(1, -1))
                else:
                    raise ValueError("Invalid feature version")
        
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                if feature_version == 1:
                    prediction = self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
                elif feature_version == 2:
                    prediction = self.model.predict(np.array(get_features_v2(board)).reshape(1, -1))
                elif feature_version == 3:
                    prediction = self.model.predict(np.array(get_features_v3(board)).reshape(1, -1))
                else:
                    raise ValueError("Invalid feature version")
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        return best_move_uci
        
    def alpha_beta(self, board, depth, alpha, beta, maximizing_player, feature_version=1):
        if depth == 0:
            if feature_version == 1:
                return self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
            elif feature_version == 2:
                return self.model.predict(np.array(get_features_v2(board)).reshape(1, -1))
            elif feature_version == 3:
                return self.model.predict(np.array(get_features_v3(board)).reshape(1, -1))
            #return self.model.predict(np.array(get_features_v1(board)).reshape(1, -1))
        if maximizing_player:
            best_score = -np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.alpha_beta(board, depth - 1, alpha, beta, False, feature_version=feature_version)
                board.pop()
                best_score = max(score, best_score) 
                alpha = max(alpha, score)
                if beta <= alpha: # alpha cut-off
                    break
            return best_score
        else:
            best_score = np.inf
            for move in board.legal_moves:
                board.push(move)
                score = self.alpha_beta(board, depth - 1, alpha, beta, True, feature_version=feature_version)
                board.pop()
                best_score = min(score, best_score)
                beta = min(beta, score)
                if beta <= alpha: # beta cut-off
                    break
            return best_score
        
    def get_best_move_alpha_beta(self, board, depth, feature_version=1):
        best_move_uci = None
        if board.turn == chess.WHITE:
            best_move_prediction = 0
            for move in board.legal_moves:
                board.push(move)
                prediction = self.alpha_beta(board, depth, -np.inf, np.inf, False, feature_version=feature_version)
                if prediction > best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()
        else:
            best_move_prediction = 1
            for move in board.legal_moves:
                board.push(move)
                prediction = self.alpha_beta(board, depth, -np.inf, np.inf, True, feature_version=feature_version)
                if prediction < best_move_prediction:
                    best_move_prediction = prediction
                    best_move_uci = move.uci()
                board.pop()

        if best_move_uci is None:
            # pick random legal move
            best_move_uci = board.legal_moves[0].uci()
        return best_move_uci
    