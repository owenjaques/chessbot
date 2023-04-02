import chess
import chess.pgn
import numpy as np

class Boardprocessing:

    def __init__(self, board, board_style="V1"):
        self.board = chess.Board(board)
        self.board_image = [0]*64
        self.board_style = board_style
        self.update_board()
        self.get_board_image()

    def update_board(self):
        if self.board_style == "V1":
            self.update_board_V1()
        elif self.board_style == "V2":
            self.update_board_V2()
        else:
            print("why")
        
    def update_board_V1(self):
        fen_split = list(self.board.fen().split()[0])
        turn_marker = -1
        if not self.board.turn:
            turn_marker = 1
        count = 0
        for each in fen_split:
            if each == 'p':
                self.board_image[count] = 3*turn_marker
            elif each == 'P':
                self.board_image[count] = -5*turn_marker
            elif each == 'b':
                self.board_image[count] = 11*turn_marker
            elif each == 'B':
                self.board_image[count] = -13*turn_marker
            elif each == 'n':
                self.board_image[count] = 17*turn_marker
            elif each == 'N':
                self.board_image[count] = -19*turn_marker
            elif each == 'r':
                self.board_image[count] = 29*turn_marker
            elif each == 'R':
                self.board_image[count] = -31*turn_marker
            elif each == 'q':
                self.board_image[count] = 41*turn_marker
            elif each == 'Q':
                self.board_image[count] = -43*turn_marker
            elif each == 'k':
                self.board_image[count] = 101*turn_marker
            elif each == 'K':
                self.board_image[count] = -103*turn_marker
            elif each.isdigit():
                for x in range(0, int(each)-1):
                    count += 1
            if each != "/":
                count += 1
                if count > 63:
                    break

    def update_board_V2(self):
        fen_split = list(self.board.fen().split()[0])
        count = 0
        if self.board.turn:
            for each in fen_split:
                if each == 'p':
                    self.board_image[count] = 2
                elif each == 'P':
                    self.board_image[count] = 17
                elif each == 'b':
                    self.board_image[count] = 3
                elif each == 'B':
                    self.board_image[count] = 19
                elif each == 'n':
                    self.board_image[count] = 5
                elif each == 'N':
                    self.board_image[count] = 23
                elif each == 'r':
                    self.board_image[count] = 7
                elif each == 'R':
                    self.board_image[count] = 29
                elif each == 'q':
                    self.board_image[count] = 11
                elif each == 'Q':
                    self.board_image[count] = 31
                elif each == 'k':
                    self.board_image[count] = 13
                elif each == 'K':
                    self.board_image[count] = 37
                elif each.isdigit():
                    for x in range(0, int(each)-1):
                        count += 1
                if each != "/":
                    count += 1
                    if count > 63:
                        break
        else:
            for each in fen_split:
                if each == 'p':
                    self.board_image[count] = 17
                elif each == 'P':
                    self.board_image[count] = 2
                elif each == 'b':
                    self.board_image[count] = 19
                elif each == 'B':
                    self.board_image[count] = 3
                elif each == 'n':
                    self.board_image[count] = 23
                elif each == 'N':
                    self.board_image[count] = 5
                elif each == 'r':
                    self.board_image[count] = 29
                elif each == 'R':
                    self.board_image[count] = 7
                elif each == 'q':
                    self.board_image[count] = 31
                elif each == 'Q':
                    self.board_image[count] = 11
                elif each == 'k':
                    self.board_image[count] = 37
                elif each == 'K':
                    self.board_image[count] = 13
                elif each.isdigit():
                    for x in range(0, int(each)-1):
                        count += 1
                if each != "/":
                    count += 1
                    if count > 63:
                        break


    def get_board_image(self):
        return np.array(self.board_image).reshape(1,-1)


if __name__ == '__main__':
    board = chess.Board()
    model_input = Boardprocessing(board, "V1").get_board_image()
    print(model_input)
    print(model_input.shape)
