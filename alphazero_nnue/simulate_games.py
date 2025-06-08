import random
from nn import NeuralNetwork, load_model
from mcts import mcts
from board import Board
import time
import globals
import math
from multiprocessing import Pool

nn_name_control = input("Enter the control's model name: ")
nn_name_experimental = input("Enter the experimental's model name: ")
control_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_control + '.pth')
experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_experimental + '.pth')
control_model.eval()
experimental_model.eval()

draws, control_wins, experimental_wins = 0, 0, 0
def simulate_game(identifier):
    # global draws, control_wins, experimental_wins
    control_player = random.randint(0, 1)
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000
    board = Board(initial_player_board, initial_opponent_board, current_player)
    while not board.game_ends():
        if current_player == control_player:
            (player_board, opponent_board) = mcts(board, control_model, False, True)
        else:
            (player_board, opponent_board) = mcts(board, experimental_model, False, True)
        current_player = 1 - current_player
        board = Board(player_board, opponent_board, current_player)

    winner = board.get_winner()
    current_is_control = current_player == control_player
    if winner == 0:
        # draws += 1
        return 0
    elif (current_is_control and winner == 1) or (not current_is_control and winner == -1):
        # control_wins += 1
        return -1
    else:
        # experimental_wins += 1
        return 1

# 0 - Black
# 1 - White
num_games = int(input("Enter the number of games: "))
start = time.perf_counter()
current_game = 0

for result in Pool().map(simulate_game, range(num_games)):
    if result == 0:
        draws += 1
    elif result == -1:
        control_wins += 1
    else:
        experimental_wins += 1
    current_game += 1
    if current_game % 100 == 0:
        print("At Game #" + str(current_game))

num_games -= draws
experimental_WR = experimental_wins / num_games
probability = 1.0 * experimental_wins / num_games
error = 1.96 * math.sqrt(probability * (1 - probability) / num_games)
lower_bound = 100.0 * (probability - error)
upper_bound = 100.0 * (probability + error)
print("Number of Draws:", draws)
print("Number of Control Wins:", control_wins)
print("Number of Experimental Wins:", experimental_wins)
print("Experimental WR: " + str(experimental_WR * 100.0) + "%")
print("95% Confidence Interval: [" + str(lower_bound) + "%, " + str(upper_bound) + "%]")
end = time.perf_counter()
print("Execution time (s):", end - start)