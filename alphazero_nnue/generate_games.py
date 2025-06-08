import random

from nn import NeuralNetwork, load_model
from mcts import mcts
from board import Board
import time
import globals
import math
from multiprocessing import Pool

# think about letting the bot play older generations if training is unstable
# features.bin -> file containing board states in a binary format
# labels.txt -> file containing (policy, avg(MCTS averaged value, win/loss overall)) (65 numbers)
nn_name_control = input("Enter the control's model name: ")
nn_name_experimental = input("Enter the experimental's model name: ")
control_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_control + '.pth')
experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_experimental + '.pth')
control_model.eval()
experimental_model.eval()

def generate_game(identifier):
    control_player = random.randint(0, 1)
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000
    board = Board(initial_player_board, initial_opponent_board, current_player)
    game = []
    while not board.game_ends():
        game.append(board)
        if current_player == control_player:
            (player_board, opponent_board) = mcts(board, control_model)
        else:
            (player_board, opponent_board) = mcts(board, experimental_model)

        current_player = 1 - current_player
        board = Board(player_board, opponent_board, current_player)

    # end board not included
    # game.append(board)
    game_winner = board.get_winner()
    return_game = []
    # weight full games more in later generations (make epsilon closer to 1)
    epsilon = 0.5
    for position in game:
        position_string = format(position.player_board, '064b') + format(position.opponent_board, '064b')
        policy = position.get_full_policy()
        winner = game_winner * (1 if position.player == current_player else -1)
        value = (1 - epsilon) * (position.sum_eval / position.visited_count) + epsilon * winner
        return_game.append((position_string, policy, value))

    return return_game

num_games = int(input("Enter the number of games: "))
start = time.perf_counter()

open('datasets/features.bin', 'wb')
open('datasets/labels.txt', 'w')
for game in Pool().map(generate_game, range(num_games)):
    for position in game:
        position_string, policy, value = position
        with open('datasets/features.bin', 'ab') as f:
            f.write(int(position_string, 2).to_bytes(16, byteorder='big'))
        with open('datasets/labels.txt', 'a') as f:
            f.write(' '.join(map(str, policy)) + ' ' + str(value) + '\n')