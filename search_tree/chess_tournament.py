# run a chess tournament with a given number of chess playing agents and a given number of rounds
# the agents are randomly paired up and play a game of chess
# the agents are then ranked by their win percentage
# the purpose is to analyze the performance of the agents not to train them
# the agents are not trained during the tournament

import chess
import chess.pgn
import chess.engine
import chess.svg
import random
import numpy as np
import os
import sys
import time
import datetime
import argparse
import chess_tournament

# import the agents
from agents import *

# set the number of rounds
rounds = 2

# set the number of games per round
games_per_round = 2

# set the number of agents
agents = [RandomAgent(), 
            MinimaxAgent(),
            MinimaxABAgent(),
            MCTSHeapAgent(),
            MCTSHeapAgent2000and15(),
            ChessBotAgentBtfSimpleInput()
            ]

# set the number of games per agent
games_per_agent = games_per_round * len(agents) / 2

# class to store the results of the tournament
class TournamentResults:
    def __init__(self, agents):
        self.agents = agents
        self.results = {}
        for agent in agents:
            self.results[agent.name] = 0

# run a chess tournament with a given number of chess playing agents and a given number of rounds
# the agents are randomly paired up and play a game of chess
# the agents are then ranked by their win percentage
# the purpose is to analyze the performance of the agents not to train them
# the agents are not trained during the tournament
def run_tournament(agents, rounds, games_per_round):
    # create a tournament results object
    tournament_results = TournamentResults(agents)
    # loop through the rounds
    for round in range(rounds):
        # loop through the games per round
        for game in range(games_per_round):
            # randomly select two agents
            agent1 = random.choice(agents)
            agent2 = random.choice(agents)
            # make sure the agents are not the same
            while agent1 == agent2:
                agent2 = random.choice(agents)
            # play a game between the two agents
            game_results = play_game(agent1, agent2)
            # update the tournament results
            tournament_results.results[agent1.name] += game_results.results[agent1.name]

    # return the tournament results
    return tournament_results

# play a game between two agents
def play_game(agent1, agent2):
    # create a chess board
    board = chess.Board()
    # create a game results object
    game_results = TournamentResults([agent1, agent2])
    # loop through the moves
    while not board.is_game_over( claim_draw=True ) and not board.is_stalemate() and not board.is_insufficient_material() and not board.is_fivefold_repetition():
        # get the move from the first agent
        move = agent1.get_move(board)
        # make the move
        board.push(move)
        # get the move from the second agent
        move = agent2.get_move(board)
        # make the move
        board.push(move)
    # get the result of the game
    result = board.result()
    # update the game results
    if result == "1-0":
        game_results.results[agent1.name] += 1
    elif result == "0-1":
        game_results.results[agent2.name] += 1
    else:
        game_results.results[agent1.name] += 0.5
        game_results.results[agent2.name] += 0.5
    # return the game results
    return game_results

# print the tournament results
def print_tournament_results(tournament_results):
    # loop through the agents
    for agent in tournament_results.agents:
        # print the agent name and win percentage
        print(agent.name + ": " + str(tournament_results.results[agent.name] / games_per_agent))

# save the tournament results
def save_tournament_results(tournament_results):
    # create a file name
    file_name = "tournament_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
    # open the file
    file = open(file_name, "w")
    # loop through the agents
    for agent in tournament_results.agents:
        # write the agent name and win percentage to the file
        file.write(agent.name + ": " + str(tournament_results.results[agent.name] / games_per_agent) + "    ")
    # close the file
    file.close()



def main():
    # run the tournament
    tournament_results = run_tournament(agents, rounds, games_per_round)
    # print the tournament results
    print_tournament_results(tournament_results)
    # save the tournament results
    save_tournament_results(tournament_results)



if __name__ == "__main__":
    main()
