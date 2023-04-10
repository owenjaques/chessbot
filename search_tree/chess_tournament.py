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
from IPython.display import clear_output
from IPython.display import display

import matplotlib.pyplot as plt
sys.path.append('..')

# import the agents
from agents import *


# class to store the results of the tournament
class TournamentResults:
    def __init__(self, agents):
        self.agents = agents
        self.results = {}
        self.failed_games = {}
        for agent in agents:
            self.results[agent.name] = 0
            # store the results of each agent against each other agent
            for agent2 in agents:
                self.results[agent.name + " " + agent2.name] = 0

class ChessTournament():
    def __init__(self, agents, rounds, games_per_round, should_visualize=False):
        self.agents = agents
        self.rounds = rounds
        self.games_per_round = games_per_round
        self.results = None
        self.games_per_agent = self.games_per_round * len(self.agents) / 2
        self.should_visualize = should_visualize

    # run a chess tournament with a given number of chess playing agents and a given number of rounds
    # the agents are randomly paired up and play a game of chess
    # the agents are then ranked by their win percentage
    # the purpose is to analyze the performance of the agents not to train them
    # the agents are not trained during the tournament
    def run_tournament(self):
        # create a tournament results object
        tournament_results = TournamentResults(self.agents)
        # loop through the rounds
        for round in range(self.rounds):
            # loop through the games per round
            for game in range(self.games_per_round):
                # randomly select two agents
                agent1 = random.choice(self.agents)
                agent2 = random.choice(self.agents)
                # make sure the agents are not the same
                while agent1 == agent2:
                    agent2 = random.choice(self.agents)

                agent1.initialize(chess.WHITE)
                agent2.initialize(chess.BLACK)
                # play a game between the two agents
                try:
                    game_results = self.play_game(agent1, agent2)
                    # update the tournament results
                    tournament_results.results[agent1.name] += game_results.results[agent1.name]
                    tournament_results.results[agent2.name] += game_results.results[agent2.name]
                    tournament_results.results[agent1.name + " " + agent2.name] += game_results.results[agent1.name]
                    tournament_results.results[agent2.name + " " + agent1.name] += game_results.results[agent2.name]
                    # save the game results
                    self.save_progress(tournament_results)
                except:
                    print("Game failed!, moving onto next game")
                    if agent1.name in tournament_results.failed_games:
                        tournament_results.failed_games[agent1.name] += 1
                    else:
                        tournament_results.failed_games[agent1.name] = 1
                    if agent2.name in tournament_results.failed_games:
                        tournament_results.failed_games[agent2.name] += 1
                    else:
                        tournament_results.failed_games[agent2.name] = 1
                    continue

        print("Tournament over!")

        # loop through the agents
        for agent in self.agents:
            # calculate the win percentage
            win_percentage = tournament_results.results[agent.name] / (self.games_per_agent * self.rounds)
            # print the results
            print(agent.name + " win percentage: " + str(win_percentage))

        # return the tournament results
        return tournament_results

    # play a game between two agents
    def play_game(self, agent1, agent2):
        # create a chess board
        board = chess.Board()
        # create a game results object
        game_results = TournamentResults([agent1, agent2])
        print("Game started!")
        print(agent1.name + " " + agent2.name + ": ")
        # loop through the moves
        while not board.is_game_over( claim_draw=True ) and not board.is_stalemate() and not board.is_insufficient_material() and not board.is_fivefold_repetition():
            # get the move from the first agent
            move = agent1.get_move(board)
            # make the move
            board.push(move)
            if self.should_visualize:
                clear_output(wait=True)
                display(board)
                print(agent1.name + " " + agent2.name + ": ")
            # get the move from the second agent
            move = agent2.get_move(board)
            # make the move
            board.push(move)
            if self.should_visualize:
                clear_output(wait=True)
                display(board)
                print(agent1.name + " " + agent2.name + ": ")
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

        # print the game result
        print(agent1.name + " " + agent2.name + ": " + result)

        # return the game results
        return game_results
    
    # save progress
    def save_progress(self, tournament_results):
        # create a file name
        file_name = "tournament_results_" + datetime.datetime.now().strftime("%Y%m%d") + ".txt"
        # open the file
        file = open(file_name, "w")
        # loop through the agents
        for agent in tournament_results.agents:
            # write the agent name and win percentage to the file
            file.write(agent.name + ": " + str(tournament_results.results[agent.name] / self.games_per_agent) + "    ")

        for agent in tournament_results.failed_games:
            file.write(agent.name + ": " + str(tournament_results.failed_games[agent.name]) + "    ")

        # print the agaent against agent results
        for agent in tournament_results.agents:
            for agent2 in tournament_results.agents:
                file.write(agent.name + " " + agent2.name + ": " + str(tournament_results.results[agent.name + " " + agent2.name] / self.games_per_round) + "    ")

        # close the file
        file.close()

    # print the tournament results
    def print_tournament_results(self, tournament_results):
        # loop through the agents
        for agent in tournament_results.agents:
            # print the agent name and win percentage
            print(agent.name + ": " + str(tournament_results.results[agent.name] / self.games_per_agent))

    # save the tournament results
    def save_tournament_results(self, tournament_results):
        # create a file name
        file_name = "tournament_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
        # open the file
        file = open(file_name, "w")
        # loop through the agents
        for agent in tournament_results.agents:
            # write the agent name and win percentage to the file
            file.write(agent.name + ": " + str(tournament_results.results[agent.name] / self.games_per_agent) + "    ")

        # print the agaent against agent results
        for agent in tournament_results.agents:
            for agent2 in tournament_results.agents:
                file.write(agent.name + " " + agent2.name + ": " + str(tournament_results.results[agent.name + " " + agent2.name] / self.games_per_round) + "    ")


        # close the file
        file.close()


    # plot the tournament results as a heat map and save the heat map
    def plot_tournament_results(self, tournament_results):
        # create a list of the agent names
        agent_names = []
        for agent in tournament_results.agents:
            agent_names.append(agent.name)
        # create a list of the agent against agent results
        agent_results = []
        for agent in tournament_results.agents:
            agent_results.append([])
            for agent2 in tournament_results.agents:
                agent_results[-1].append(tournament_results.results[agent.name + " " + agent2.name] / self.games_per_round)
        # create a numpy array of the agent against agent results
        agent_results = np.array(agent_results)
        # plot the heat map
        fig, ax = plt.subplots()
        im = ax.imshow(agent_results)
        # set the x and y ticks
        ax.set_xticks(np.arange(len(agent_names)))
        ax.set_yticks(np.arange(len(agent_names)))
        # set the x and y tick labels
        ax.set_xticklabels(agent_names)
        ax.set_yticklabels(agent_names)
        # rotate the x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        # loop through the data dimensions and create text annotations
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                text = ax.text(j, i, agent_results[i, j],
                            ha="center", va="center", color="w")
        # set the title
        ax.set_title("Tournament Results")
        # save the heat map
        fig.tight_layout()
        plt.savefig("tournament_results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
        plt.show()




def main():
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
                MCTSAgent(),
                MCTSBtfSimple(),
                MCTSOwenSimple(),
                MCTSOwenBtfSimple(),
                ChessBotAgentBtfSimpleInput(),
                ChessBotAgentBtfSingleInput(),
                ChessBotAgentBtfTripleInput(),
                ChessBotAgentOwenSimpleInput(),
                ]
    
    tourny = ChessTournament(agents, rounds, games_per_round)

    # set the number of games per agent
    games_per_agent = games_per_round * len(agents) / 2
    # run the tournament
    tournament_results = tourny.run_tournament()
    # print the tournament results
    tourny.print_tournament_results(tournament_results)
    # save the tournament results
    tourny.save_tournament_results(tournament_results)

    # plot the tournament results
    tourny.plot_tournament_results(tournament_results)



if __name__ == "__main__":
    main()
