#!/usr/bin/env python3
import chess
import chess.svg
from stockfish import Stockfish
#This line should be replaced with whatever model we want to test.
#The model must impliment a function called get_next_move().
import a_really_bad_chess_engine


#move must be guaranteed to be a legal move
def get_winner_of_game(stockfish_engine, ML_engine):
    #initialize the board
    outcome = False
    stockfish_engine.set_position([])
    board = chess.Board()
    turn = chess.WHITE
        #check if the game is over
        #must update both board and stockfish with the move
    while(not outcome):
        if(turn == chess.WHITE):
            #get stockfish's move
            move = stockfish_engine.get_best_move()
            board.push(chess.Move.from_uci(move))
            stockfish_engine.make_moves_from_current_position([move])
            turn = not turn
            ##uncomment these lines to see chess board in browser - must refresh browser to see updated results...chessboard class may be better option.
            #with open("chess_display.html", "w") as chess_display:
            #    chess_display.write(chess.svg.board(board, size = 500))
            
        else:
            #get the ML_model's move
            move = ML_engine.get_next_move(board)
            board.push(move)
            stockfish_engine.make_moves_from_current_position([move])
            turn = not turn
            ##uncomment these lines to see chess board in browser - must refresh browser to see updated results...chessboard class may be better option.
            #with open("chess_display.html", "w") as chess_display:
            #    chess_display.write(chess.svg.board(board, size = 500))
            
        outcome = board.outcome()
    #If the winner is True, then the stockfish engine won.
    print(outcome)
    return outcome

def calculate_elo():
    pass##TODO

def main():
    #change to wherever you've installed stockfish
    stockfish = Stockfish(path = "/usr/local/bin/stockfish", parameters = {"UCI_LimitStrength" : "true"})
    games_won = []
    for elo in range(1350, 2851, 50):
        stockfish.update_engine_parameters({"UCI_Elo" : elo})
        winner = get_winner_of_game(stockfish, a_really_bad_chess_engine).winner
        games_won += [(elo, not winner)]
    print(games_won)

if __name__ == "__main__":
    main()