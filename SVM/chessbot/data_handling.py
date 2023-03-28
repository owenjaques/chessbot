import chess
import chess.pgn


def game_to_positions(game):

    board = chess.Board()
    positions = []

    for move in game.mainline_moves():

        board.push(move)
        positions.append(board.copy())

    return positions

def positions_to_vectors(positions):
        vectors = []
        for position in positions:
            vectors.append(position_to_vector(position))
        return vectors

def position_to_vector(position):
    vector = []
    for square in chess.SQUARES:
        if position.piece_type_at(square) == None:
            vector.append(0)
        elif position.piece_at(square).color == chess.WHITE:
            vector.append(position.piece_type_at(square))
        else:
            vector.append(position.piece_type_at(square) * -1)

    return vector

def result_to_label(result):
        if result == '1-0':
            return 1
    
        elif result == '0-1':
            return 0
    
        else:
            return 0.5
        
def game_to_data(game, position):
    position = position_to_vector(position)
    if game.headers["Result"] == '1-0':
        result = 1
    elif game.headers["Result"] == '0-1':
        result = 0
    else:
        result = 0.5
    return [position, result]


def pgn_to_data(pgn, game_number):
    X_all = []
    y_all = []
    count = 0

    game = chess.pgn.read_game(pgn)
    while game != None and count < game_number:
        positions = game_to_positions(game)
        for position in positions:
            X_all.append(position_to_vector(position))
            y_all.append(result_to_label(game.headers["Result"]))
        count += 1
        game = chess.pgn.read_game(pgn)

    return [X_all, y_all]

