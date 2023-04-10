#!/usr/bin/env python3

#This class is meant to estimate the elo of our chess engine.
import chess
import chess.svg
from stockfish import Stockfish
import typing
import random
#This line should be replaced with whatever model we want to test.
#The model must impliment a function called get_next_move().
#import a_really_bad_chess_engine

from IPython.display import clear_output
from IPython.display import display


#move must be guaranteed to be a legal move
def get_winner_of_game(stockfish_engine, ML_engine, display_board = False):
    #initialize the board
    outcome = False
    stockfish_engine.set_position([])
    board = chess.Board()
    turn = chess.WHITE
    #check if the game is over
    while(not outcome):
        if(turn == chess.WHITE):
            #get stockfish's move
            move = stockfish_engine.get_best_move()
            #must update both board and stockfish with the move
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
        if(display_board):
            clear_output(wait=True)
            display(board)
        outcome = board.outcome()
    #If the winner is True, then the stockfish engine won.
    print(outcome)
    return outcome

def calculate_elo( winner, loser, k = 32) -> typing.Tuple[int, int]:
    expected_winner = 1 / (1 + 10 ** ((loser - winner) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner - loser) / 400))

    new_winner = winner + k * (1 - expected_winner)
    new_loser = loser + k * (0 - expected_loser)
    return (new_winner, new_loser)

def run_elo_evaluation(our_agent, stockfish_opponent, display_board):
    #change to wherever you've installed stockfish
    games_won = []
    #starting elo
    our_elo = 1500
    #list of elos which will set the stokcfish engine's elo for each game
    elos = list(range(1000, 1501, 50))
    random.shuffle(elos)

    for elo in elos:
        stockfish_opponent.update_engine_parameters({"UCI_Elo" : elo})
        winner = get_winner_of_game(stockfish_opponent, our_agent, display_board).winner
        games_won += [(elo, not winner)]
        if winner:
            elo, our_elo = calculate_elo(elo, our_elo)
        else:
            our_elo, elo = calculate_elo(our_elo, elo)

    print(games_won)
    print(our_elo)
    return our_elo

def main():
    """
    #change to wherever you've installed stockfish
    stockfish = Stockfish(path = "/usr/local/bin/stockfish", parameters = {"UCI_LimitStrength" : "true"})
    games_won = []
    #starting elo
    our_elo = 1500
    #list of elos which will set the stokcfish engine's elo for each game
    elos = list(range(1350, 2851, 20))
    random.shuffle(elos)

    for elo in elos:
        stockfish.update_engine_parameters({"UCI_Elo" : elo})
        winner = get_winner_of_game(stockfish, a_really_bad_chess_engine).winner
        games_won += [(elo, not winner)]
        if winner:
            elo, our_elo = calculate_elo(elo, our_elo)
        else:
            our_elo, elo = calculate_elo(our_elo, elo)

    print(games_won)
    print(our_elo)
    """
    print("hello :)")

if __name__ == "__main__":
    main()