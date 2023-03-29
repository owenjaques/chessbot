# a class that represents the state of the game that can be inherited by classes for specific games
# 

import random

class GameState:
    def __init__(self):
        self.turn = 1
        self.board = None
        self.moves = None
        self.result = [None, None]

    def get_moves(self):
        return self.moves

    def get_random_move(self):
        if self.moves is None:
            return None
        return random.choice(self.moves)

    def make_move(self, move):
        pass

    def is_over(self):
        return self.result is not None

    def get_result(self, player):
        return self.result[player]

    def copy(self):
        pass

    def __repr__(self):
        return str(self.board)
    
    def __str__(self):
        return str(self.board)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return str(self.board) == str(other.board)
    
    def __ne__(self, other):
        return str(self.board) != str(other.board)
    

# a GameState class for for chess that inherits from the GameState class
# should use the python-chess library
#

class ChessState(GameState):
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.moves = list(self.board.legal_moves)
        self.result = None

    def make_move(self, move):
        self.board.push(move)
        self.moves = list(self.board.legal_moves)
        self.turn = 1 if self.turn == 2 else 2

    def get_moves(self):
        return self.moves
    
    def get_random_move(self):
        if self.moves is None:
            return None
        return random.choice(self.moves)

    def is_over(self):
        if self.board.is_game_over():
            self.result = self.board.result()
            return True
        return False

    def get_result(self, player):
        if self.result is not None:
            if self.result == '1-0':
                return 1 if player == 1 else 0
            elif self.result == '0-1':
                return 1 if player == 2 else 0
            else:
                return 0.5
        return None

    def copy(self):
        return ChessState(self.board.copy())

    def __repr__(self):
        return str(self.board)
    
    def __str__(self):
        return str(self.board)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return str(self.board) == str(other.board)
    
    def __ne__(self, other):
        return str(self.board) != str(other.board)
    

# a GameState class for for tic-tac-toe that inherits from the GameState class
#

class TicTacToeState(GameState):
    def __init__(self, board, turn):
        super().__init__()
        self.board = board
        self.turn = turn
        self.moves = self.get_moves()
        self.result = None

    def get_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def get_random_move(self):
        if self.moves is None:
            return None
        return random.choice(self.moves)

    def make_move(self, move):
        self.board[move[0]][move[1]] = self.turn
        self.moves = self.get_moves()
        self.turn = 1 if self.turn == 2 else 2

    def is_over(self):
        if self.result is not None:
            return True
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] != 0:
                self.result = self.board[i][0]
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] != 0:
                self.result = self.board[0][i]
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != 0:
            self.result = self.board[0][0]
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != 0:
            self.result = self.board[0][2]
            return True
        if len(self.moves) == 0:
            self.result = 0
            return True
        return False

    def get_result(self, player):
        if self.result is not None:
            if self.result == player:
                return 1
            elif self.result == 0:
                return 0.5
            else:
                return 0
        return None

    def copy(self):
        return TicTacToeState([row[:] for row in self.board], self.turn)

    def __repr__(self):
        return str(self.board)
    
    def __str__(self):
        return str(self.board)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return str(self.board) == str(other.board)
    
    def __ne__(self, other):
        return str(self.board) != str(other.board)
    

# a GameState class for for connect four that inherits from the GameState class
#

class ConnectFourState(GameState):
    def __init__(self, board, turn):
        super().__init__()
        self.board = board
        self.turn = turn
        self.moves = self.get_moves()
        self.result = None

    def get_moves(self):
        moves = []
        for i in range(7):
            if self.board[0][i] == 0:
                moves.append(i)
        return moves
    
    def get_random_move(self):
        if self.moves is None:
            return None
        return random.choice(self.moves)

    def make_move(self, move):
        for i in range(5, -1, -1):
            if self.board[i][move] == 0:
                self.board[i][move] = self.turn
                break
        self.moves = self.get_moves()
        self.turn = 1 if self.turn == 2 else 2

    def is_over(self):
        if self.result is not None:
            return True
        for i in range(6):
            for j in range(4):
                if self.board[i][j] == self.board[i][j+1] == self.board[i][j+2] == self.board[i][j+3] and self.board[i][j] != 0:
                    self.result = self.board[i][j]
                    return True
        for i in range(3):
            for j in range(7):
                if self.board[i][j] == self.board[i+1][j] == self.board[i+2][j] == self.board[i+3][j] and self.board[i][j] != 0:
                    self.result = self.board[i][j]
                    return True
        for i in range(3):
            for j in range(4):
                if self.board[i][j] == self.board[i+1][j+1] == self.board[i+2][j+2] == self.board[i+3][j+3] and self.board[i][j] != 0:
                    self.result = self.board[i][j]
                    return True
        for i in range(3):
            for j in range(3, 7):
                if self.board[i][j] == self.board[i+1][j-1] == self.board[i+2][j-2] == self.board[i+3][j-3] and self.board[i][j] != 0:
                    self.result = self.board[i][j]
                    return True
        if len(self.moves) == 0:
            self.result = 0
            return True
        return False
    
    def get_result(self, player):
        if self.result is not None:
            if self.result == player:
                return 1
            elif self.result == 0:
                return 0.5
            else:
                return 0
        return None
    
    def copy(self):
        return ConnectFourState([row[:] for row in self.board], self.turn)
    
    def __repr__(self):
        return str(self.board)
    
    def __str__(self):
        return str(self.board)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return str(self.board) == str(other.board)
    
    def __ne__(self, other):
        return str(self.board) != str(other.board)
    

# a GameState class for for checkers that inherits from the GameState class
#

class CheckersState(GameState):
    def __init__(self, board, turn):
        super().__init__()
        self.board = board
        self.turn = turn
        self.moves = self.get_moves()
        self.result = None

    def get_moves(self):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == self.turn:
                    if self.turn == 1:
                        if i > 0 and j > 0 and self.board[i-1][j-1] == 0:
                            moves.append(((i, j), (i-1, j-1)))
                        if i > 0 and j < 7 and self.board[i-1][j+1] == 0:
                            moves.append(((i, j), (i-1, j+1)))
                    else:
                        if i < 7 and j > 0 and self.board[i+1][j-1] == 0:
                            moves.append(((i, j), (i+1, j-1)))
                        if i < 7 and j < 7 and self.board[i+1][j+1] == 0:
                            moves.append(((i, j), (i+1, j+1)))
        return moves
    
    def get_random_move(self):
        if self.moves is None:
            return None
        return random.choice(self.moves)

    def make_move(self, move):
        self.board[move[1][0]][move[1][1]] = self.board[move[0][0]][move[0][1]]
        self.board[move[0][0]][move[0][1]] = 0
        self.moves = self.get_moves()
        self.turn = 1 if self.turn == 2 else 2

    def is_over(self):
        if self.result is not None:
            return True
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1:
                    if i < 7 and j > 0 and self.board[i+1][j-1] == 0:
                        return False
                    if i < 7 and j < 7 and self.board[i+1][j+1] == 0:
                        return False
                if self.board[i][j] == 2:
                    if i > 0 and j > 0 and self.board[i-1][j-1] == 0:
                        return False
                    if i > 0 and j < 7 and self.board[i-1][j+1] == 0:
                        return False
        self.result = 0
        return True
    
    def get_result(self, player):
        if self.result is not None:
            if self.result == player:
                return 1
            elif self.result == 0:
                return 0.5
            else:
                return 0
        return None
    
    def copy(self):
        return CheckersState([row[:] for row in self.board], self.turn)
    
    def __repr__(self):
        return str(self.board)
    
    def __str__(self):
        return str(self.board)
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return str(self.board) == str(other.board)
    
    def __ne__(self, other):
        return str(self.board) != str(other.board)
    

