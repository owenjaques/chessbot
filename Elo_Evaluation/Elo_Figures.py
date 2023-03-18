#!/usr/bin/env python3

#this script takes output from play_games.py and and creates a plot showing the elo as games are played.

import play_games
import matplotlib.pyplot as plt

#output from play_games.py with the elo step size set to 20
elos = [(1970, False), (2310, False), (2770, False), (2810, False), (2730, False), (2750, False), (2270, False), (1990, False), (2350, False), (2610, False), (1490, True), (1870, True), (2390, False), (1810, False), (2470, False), (2170, False), (2530, False), (1530, True), (2690, False), (2210, False), (2550, False), (2090, False), (1370, True), (1430, True), (2630, False), (2430, False), (1850, False), (2570, False), (2030, False), (1730, True), (1470, True), (2250, False), (1650, True), (1390, True), (2710, False), (2510, False), (1950, True), (1930, False), (1670, True), (1830, True), (1770, False), (2150, False), (1410, True), (1350, True), (2370, False), (1910, False), (1590, True), (2850, False), (1790, False), (1510, True), (2450, False), (2050, True), (1630, True), (2650, False), (2330, False), (1550, True), (1890, False), (2070, False), (1450, True), (1710, True), (1610, True), (1750, True), (2590, False), (2830, False), (1570, True), (2490, False), (2290, False), (2410, False), (2130, False), (1690, True), (2670, False), (2010, False), (2790, False), (2190, False), (2230, False), (2110, False)]

#create subsets of the games.  Add the next game played to the subset each iteration.
our_elos = []
for i in range(0, len(elos)):
    elos_subset = elos[0:i]
    #calculate the elo for that subset of games
    #changing this starting value can help identify the actual elo
    our_elo = 1700
    for elo in elos_subset:
        if(not elo[1]):
            elo, our_elo = play_games.calculate_elo(elo[0], our_elo)
        else:
            our_elo, elo = play_games.calculate_elo(our_elo, elo[0])
    our_elos += [our_elo]

#To get a final elo estimation, we could do something like an exponential moving average, or even just average the results once we pick a good starting elo.
plt.plot(our_elos)
plt.xlabel("Number of Games Played")
plt.ylabel("Elo Rating")
plt.title("How Elo Rating Changes with Number of Games Played")
plt.show()