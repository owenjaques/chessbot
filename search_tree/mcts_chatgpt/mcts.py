import random
import chess
from datetime import datetime, timedelta
import math


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
            child = MCTSAgent.Node(state, move, self)
            self.untried_actions.remove(move)
            self.children.append(child)
            return child

    def __init__(self, time_limit=10):
        self.time_limit = time_limit

    def get_best_move(self, board):
        root = MCTSAgent.Node(board)
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=self.time_limit):
            node = root
            state = board.copy()

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(math.sqrt(2))

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
