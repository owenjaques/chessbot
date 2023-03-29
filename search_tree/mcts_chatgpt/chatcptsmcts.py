import chess
import math
import random


class Node:
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def add_child(self, move, child_state):
        child = Node(child_state, move, self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))


def UCB1(node):
    return node.wins / node.visits + 1.41 * math.sqrt(math.log(node.parent.visits) / node.visits)


def expand(node, child):
    move = child.move
    state = child.state
    child_node = node.add_child(move, state)
    return child_node


def UCT(node):
    if node.visits == 0:
        return node

    UCB1s = []
    for child in node.children:
        if child.visits == 0:
            return expand(node, child)
        else:
            UCB1s.append(UCB1(child))

    if not UCB1s:
        # node is a leaf node, return it
        return node

    return node.children[UCB1s.index(max(UCB1s))]


def play_game():
    board = chess.Board()
    root = Node(board)

    if board.is_game_over():
        return None

    for i in range(1000):
        node = root
        board = node.state.copy()

        # Selection
        while node.fully_expanded():
            node = UCT(node)
            move = node.move
            if board.is_legal(move):
                board.push(move)
            else:
                board.push(list(board.legal_moves)[0])

        # Expansion
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)
            child = node.add_child(move, board)

            # Simulation
            while not board.is_game_over():
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
                board.push(move)

            # Backpropagation
            result = 1 if board.result() == "1-0" else 0 if board.result() == "1/2-1/2" else -1
            while child is not None:
                child.update(result)
                child = child.parent

        # Leaf node
        else:
            result = 0 if board.result() == "1/2-1/2" else 1
            while node is not None:
                node.update(result)
                node = node.parent

    if len(root.children) > 0:
        return max(root.children, key=lambda child: child.visits).move
    else:
        return None


if __name__ == "__main__":
    print(play_game())