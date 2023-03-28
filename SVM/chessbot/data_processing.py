import chess
import chess.pgn
import numpy as np

# feature_v1 64 features + 1 label (win/draw/loss) = 65 features
# feature_v2 2 features + 1 label (win/draw/loss) = 3 features

from feature_extraction.feature_v1 import get_features as get_features_v1
from feature_extraction.feature_v2 import get_features as get_features_v2

def get_data_from_pgn(pgn_path, feature_version=1, num_data=None):
    """
    Returns X, y, where X is a list of features and y is a list of labels.
    """
    X = []
    y = []
    with open(pgn_path) as pgn:
        while True:
            if num_data is not None and len(X) >= num_data:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            # skip if game is a draw
            if game.headers["Result"] == "1/2-1/2":
                continue
            for move in game.mainline_moves():
                board.push(move)
                if feature_version == 1:
                    features = get_features_v1(board)
                elif feature_version == 2:
                    features = get_features_v2(board)
                else:
                    raise ValueError("Invalid feature version")
                label = 1 if game.headers["Result"] == "1-0" else 0
                
                X.append(features)
                y.append(label)
    return X, y

# test
'''X, y = get_data_from_pgn("data/lichess_elite.pgn", feature_version=1, num_data=10000)
print(X[0], y[0])

from sklearn import svm

model = svm.SVC(kernel="linear", C=1)
model.fit(X, y)
print("fitting done")'''

'''# test 2
X, y = get_data_from_pgn("data/lichess_elite.pgn", feature_version=2, num_games=10)
print(X[0], y[0])'''