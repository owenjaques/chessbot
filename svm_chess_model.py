import pandas as pd
from sklearn import svm
import pickle
import chess.pgn
import chess.svg
import random
import typing

# Load the data
pgn = open("./data/lichess_db_standard_rated_2013-01.pgn")

# shrink data to 1000 games
def scan_games(pgn, num_games=1000):
    for i in range(num_games):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        yield game

# for given game in the pgn file turn it into a vector of featutres [board state, move]
# representing the board state and the move that was made
def game_to_vectors(game: chess.pgn.Game):
    board = game.board()
    vectors = []
    for move in game.mainline_moves():
        board.push(move)
        vectors.append((board_to_vector(board), move.uci()))
    return vectors

# turn a board state into a vector of features
def board_to_vector(board):
    vector = []
    for i in range(64):
        piece = board.piece_at(i)
        if piece is None:
            vector.append(0)
        else:
            vector.append(piece.piece_type * piece.color)
    return vector


# train the model
def train_model(num_games=1000):
    vectors = []
    for i, game in enumerate(scan_games(pgn, num_games)):
        '''if i % 1000 == 0:
            print(i)'''
        vectors += game_to_vectors(game)
    X = [v[0] for v in vectors]
    y = [v[1] for v in vectors]
    clf = svm.SVC()
    clf.fit(X, y)
    pickle.dump(clf, open("svm_chess_model.sav", "wb"))

#train_model()


# given a board state, return the best move
def next_move(board, model) -> typing.Tuple[chess.Move, bool]:
    move = model.predict([board_to_vector(board)])[0]
    # turn move into a chess.Move object
    #check if move is legal
    if chess.Move.from_uci(move) in board.legal_moves:
        return (chess.Move.from_uci(move), True)
    else:
        #return random legal move
        #print("predicting illegal move")
        #print("giving random legal move")
        return (random.choice(list(board.legal_moves)), False)


def plot_training_time():
    import matplotlib.pyplot as plt
    import time
    game_count = []
    times = []
    for i in range(1, 100):
        '''if i*10 % 1000 == 0:
            print(i*10)'''
        start = time.time()
        train_model(i*10)
        end = time.time()
        game_count.append(i*10)
        times.append(end - start)
    plt.plot(game_count, times)
    plt.xlabel("Games")
    plt.ylabel("Time (s)")
    plt.show()


#plot_training_time()


#play game with svm model and print rate of legal moves
def legal_move_rate():
    legal_moves = 0
    total_moves = 0

    while legal_moves < 10:
        game = chess.pgn.Game()
        board = game.board()
        game.setup(board)
        node = game

        while not board.is_game_over():
            clf = pickle.load(open("svm_chess_model.sav", "rb"))
            move, legal = next_move(board, clf)
            node = node.add_variation(move)
            board.push(move)
            if legal:
                legal_moves += 1
            total_moves += 1

    print(legal_moves / total_moves)
    return legal_moves / total_moves

#legal_move_rate()



def plot_legal_move_rate():
    import matplotlib.pyplot as plt
    import time
    game_count = []
    rates = []
    for i in range(1, 10):
        train_model(i*100)
        game_count.append(i*100)
        rate = legal_move_rate()
        rates.append(rate)
    plt.plot(game_count, rates)
    plt.xlabel("Games")
    plt.ylabel("Legal Move Rate")
    plt.show()

plot_legal_move_rate()

