import threading
import random
import chess
from datetime import datetime, timedelta
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras


"""
MODELS BY CHATGPT
To test: Which model is better?
- Move predictor model
- Score predictor model

To test: Where is the best place to use the model?
- In the selection phase
- In the expansion phase
- In the simulation phase

To test: What version of MCTS is better?
- MCTS_CNN
- MCTS_MCTS_CNN_multi_thread
- MCTSAgent
- MCTS123
"""

# uses a CNN to evaluate the board state and MCTS to find the best move
class MCTS_CNN:
    class Node:
        def __init__(self, state, move=None, parent=None):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0
            self.untried_actions = list(state.legal_moves)

        def is_fully_expanded(self):
            return not self.untried_actions

        def best_child(self, exploration_constant, cnn):
            best_score = -1
            best_child = None
            for child in self.children:
                exploit = child.value / child.visits
                explore = exploration_constant * \
                    cnn.predict(child.state.board())[0][0]
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def add_child(self, move, state):
            child = MCTS_CNN.Node(state, move, self)
            self.untried_actions.remove(move)
            self.children.append(child)
            return child

    def __init__(self, time_limit=5, cnn_model=None):
        self.time_limit = time_limit
        self.cnn = cnn_model

    def get_best_move(self, board):
        root = MCTS_CNN.Node(board)
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=self.time_limit):
            node = root
            state = board.copy()

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(math.sqrt(2), self.cnn)

                state.push(node.move)

            # Expansion
            if node.untried_actions:
                move = random.choice(node.untried_actions)
                state.push(move)
                node = node.add_child(move, state)

            # Simulation
            while not state.is_game_over():
                move = random.choice(list(state.legal_moves))
                state.push(move)

            # Backpropagation
            result = self.get_result(state)
            while node is not None:
                node.visits += 1
                node.value += result
                node = node.parent

        return max(root.children, key=lambda c: c.visits).move

    def get_result(self, board):
        if board.is_checkmate():
            if board.turn:
                return -1
            else:
                return 1
        else:
            return 0
        

class Board(chess.Board):
    def board(self):
        board_matrix = np.zeros((8, 8, 6), dtype=np.uint8)
        piece_map = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                     'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
        for i in range(8):
            for j in range(8):
                piece = self.piece_at(chess.square(i, j))
                if piece is not None:
                    board_matrix[i][j][piece_map[piece.symbol()]-1] = 1
        return board_matrix
    
    def __hash__(self):
        return hash(self.fen())
    
import random


# uses a model to predict best moves in rollout and MCTS with UCT to find the best move 
class MCTS_CNN_multi_thread:
    class Node:
        def __init__(self, state, move=None, parent=None):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0
            self.untried_actions = list(state.legal_moves)

        def is_fully_expanded(self):
            return not self.untried_actions

        def best_child(self, exploration_constant):
            best_score = -1
            best_child = None
            for child in self.children:
                exploit = child.value / child.visits
                explore = exploration_constant * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def add_child(self, move, state):
            child = MCTS_CNN_multi_thread.Node(state, move, self)
            self.untried_actions.remove(move)
            self.children.append(child)
            return child

    def __init__(self, time_limit=5, num_threads=1):
        self.time_limit = time_limit
        self.num_threads = num_threads
        self.model = None
        self.exploration_constant = math.sqrt(2)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def get_best_move(self, board):
        root = MCTS_CNN_multi_thread.Node(board)
        start_time = datetime.now()

        while datetime.now() - start_time < timedelta(seconds=self.time_limit):
            threads = []
            for i in range(self.num_threads):
                thread = threading.Thread(target=self._simulate, args=(root,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        return max(root.children, key=lambda c: c.visits).move

    def _simulate(self, root):
        node = root
        state = node.state.copy()

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_constant)
            state.push(node.move)

        # Expansion
        if node.untried_actions:
            move = random.choice(node.untried_actions)
            state.push(move)
            node = node.add_child(move, state)

        # Simulation
        if not state.is_game_over():
            board_image = self._get_board_image(state)
            predictions = self.model.predict(np.array([board_image]))
            legal_moves = list(state.legal_moves)
            mask = np.ones(len(legal_moves))
            for i, move in enumerate(legal_moves):
                if move not in node.untried_actions:
                    mask[i] = 0
            masked_predictions = predictions[0] * mask
            move_index = np.argmax(masked_predictions)
            move = legal_moves[move_index]
            state.push(move)

        # Backpropagation
        result = self._get_result(state)
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def _get_board_image(self, board):
        pieces = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
        board_image = np.zeros((8, 8, 6), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i, j))
                if piece is not None:
                    board_image[i][j][pieces[piece.symbol()]-1] = 1
        return board_image
    
    def _get_result(self, board):
        if board.is_checkmate():
            if board.turn:
                return -1
            else:
                return 1
        else:
            return 0
        

    
import random
import chess
from datetime import datetime, timedelta
import math
import concurrent.futures


# uses Multi-threading to speed up MCTS with UCT to find the best move
class MCTSAgent:
    class Node:
        def __init__(self, state, move=None, parent=None):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0
            self.untried_actions = list(state.legal_moves)

        def is_fully_expanded(self):
            return not self.untried_actions

        def best_child(self, exploration_constant):
            epsilon = 1e-6
            best_score = -1
            best_child = None
            for child in self.children:
                if child.visits == 0:
                    return 0
                exploit = child.value / child.visits
                explore = exploration_constant * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def add_child(self, move, state):
            child = MCTSAgent.Node(state, move, self)
            self.untried_actions.remove(move)
            self.children.append(child)
            return child

    def __init__(self, time_limit=20, num_workers=4):
        self.time_limit = time_limit
        self.num_workers = num_workers

    def get_best_move(self, board):
        root = MCTSAgent.Node(board)
        start_time = datetime.now()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            while datetime.now() - start_time < timedelta(seconds=self.time_limit):
                # Select the most promising leaf node
                node = root
                state = board.copy()
                while not node.is_fully_expanded() and node.children:
                    node = node.best_child(math.sqrt(2))
                    state.push(node.move)

                # Expand the leaf node
                if node.untried_actions:
                    move = random.choice(node.untried_actions)
                    state.push(move)
                    node = node.add_child(move, state)

                # Simulate games from the leaf node
                futures = []
                for _ in range(self.num_workers):
                    future = executor.submit(self.simulate_game, state.copy())
                    futures.append(future)

                # Wait for the results or time out
                try:
                    results = [future.result(timeout=self.time_limit)
                            for future in futures]
                except concurrent.futures.TimeoutError:
                    # Cancel any unfinished futures and re-raise the exception
                    for future in futures:
                        future.cancel()
                    raise

                # Backpropagate the results
                for result in results:
                    node = result["node"]
                    result = result["result"]
                    while node is not None:
                        node.visits += 1
                        node.value += result
                        result = -result  # alternate the sign of the result
                        node = node.parent

        return max(root.children, key=lambda c: c.visits).move


    def simulate_game(self, board):
        node = MCTSAgent.Node(board)
        while not node.is_fully_expanded() and node.children:
            node = node.best_child(math.sqrt(2))
            board.push(node.move)

        if node.untried_actions:
            move = random.choice(node.untried_actions)
            board.push(move)
            node = node.add_child(move, board)

        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)

        result = self.get_result(board)
        return {"node": node, "result": result}

    def get_result(self, board):
        if board.is_checkmate():
            if board.turn:
                return -1
            else:
                return 1
        else:
            return 0


# this one will be interesting to test with various models and time limits
class MCTS123:
    class Node:
        def __init__(self, state, move=None, parent=None):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0
            self.untried_actions = list(state.legal_moves)

        def is_fully_expanded(self):
            return not self.untried_actions

        def best_child(self, exploration_constant):
            best_score = -1
            best_child = None
            for child in self.children:
                if child.visits == 0:
                    return 0
                exploit = child.value / child.visits
                explore = exploration_constant * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def add_child(self, move, state):
            child = MCTS123.Node(state, move, self)
            self.untried_actions.remove(move)
            self.children.append(child)
            return child

    def __init__(self, time_limit=20, num_workers=4, move_model=None, score_model=None):
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.move_model = move_model
        self.score_model = score_model

    def get_best_move(self, board):
        root = MCTS123.Node(board)
        start_time = datetime.now()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            while datetime.now() - start_time < timedelta(seconds=self.time_limit):
                # Select the most promising leaf node
                node = root
                state = board.copy()
                while not node.is_fully_expanded() and node.children:
                    node = node.best_child(math.sqrt(2))
                    state.push(node.move)

                # problem: may be too slow to make it worth it
                # needs to be tested with and without model
                # Expand the leaf node
                # with score_model evaluating the leaf node and expand highest score
                if node.untried_actions and self.score_model is not None:
                    scores = []
                    for move in node.untried_actions:
                        model_input = state.copy().push(move)
                        scores.append(self.score_model.predict(model_input))
                    move = node.untried_actions[scores.index(max(scores))]
                    state.push(move)
                    node = node.add_child(move, state)
                elif node.untried_actions:
                    move = random.choice(node.untried_actions)
                    state.push(move)
                    node = node.add_child(move, state)


                # Simulate games from the leaf node
                futures = []
                for _ in range(self.num_workers):
                    future = executor.submit(self.simulate_game, state.copy())
                    futures.append(future)

                # Wait for the results or time out
                try:
                    results = [future.result(timeout=self.time_limit)
                               for future in futures]
                except concurrent.futures.TimeoutError:
                    # Cancel any unfinished futures and re-raise the exception
                    for future in futures:
                        future.cancel()
                    raise

                # Backpropagate the results
                for result in results:
                    node = result["node"]
                    result = result["result"]
                    while node is not None:
                        node.visits += 1
                        node.value += result
                        result = -result  # alternate the sign of the result
                        node = node.parent

        return max(root.children, key=lambda c: c.visits).move

    def simulate_game(self, board):
        node = MCTS123.Node(board)
        while not node.is_fully_expanded() and node.children:
            node = node.best_child(math.sqrt(2))
            board.push(node.move)

        # maybe same issue as above? needs testing
        # simulate with move predictor model or random choice
        if self.move_model is not None and node.untried_actions:
            model_input = board.copy().push(move)
            moves = self.move_model.predict(model_input)
            move = None
            for each in moves:
                if each in node.untried_actions:
                    move = each
                    break
            if move is None:
                move = random.choice(node.untried_actions)
            board.push(move)
            node = node.add_child(move, board)
        elif node.untried_actions:
            move = random.choice(node.untried_actions)
            board.push(move)
            node = node.add_child(move, board)

        # simulate with random choice until game over or draw 
        # maybe use the move predictor mode here?
        # im not sure why the one above is different
        # it adds a node, but why there and not here?
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)

        result = self.get_result(board)
        return {"node": node, "result": result}

    def get_result(self, board):
        if board.is_checkmate():
            if board.turn:
                return -1
            else:
                return 1
        else:
            return 0

