import matplotlib.pyplot as plt
import numpy as np
import math

games = {
    (1, 2): (40, 60),
    (2, 3): (35, 65),
    (3, 4): (30, 70)
}

xpoints = [1]
ypoints = [0]
generation = 1
elo = 0
# doing correct statistical analysis on this is quite hard
# sum_variance = 0
while (generation, generation + 1) in games:
    win, lose = games[(generation, generation + 1)]
    num_games = win + lose
    p = win / num_games
    elo_gain = 400 * math.log10(1 / p - 1)
    elo += elo_gain
    generation += 1
    xpoints.append(generation)
    ypoints.append(elo)

plt.plot(xpoints, ypoints, marker = 's')
plt.xlabel("Generation")
plt.ylabel("ELO")
plt.title("ELO vs. Generation")
plt.show()