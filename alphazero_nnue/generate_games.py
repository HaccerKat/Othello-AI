import random
from nn import NeuralNetwork, load_model
from mcts import mcts
from board import Board
from board_helper import horizontal_mirror_image, rot_90_cw
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
    # prevent early game positions from flooding the dataset
    p = 3125
    while not board.game_ends():
        if random.randint(0, p) == p:
            game.append(board)

        p //= 5
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
        policy = position.get_full_policy()
        if policy[0] == -1:
            # indicates a skip turn and not to add in dataset
            continue
        winner = game_winner * (1 if position.player == current_player else -1)
        value = (1 - epsilon) * (position.sum_eval / position.visited_count) + epsilon * winner
        # debugging experiment
        # value = ((1 - epsilon) * (position.sum_eval / position.visited_count) + epsilon * winner) * -1
        return_game.append((position.player_board, position.opponent_board, policy, value))

    return return_game

num_games = int(input("Enter the number of games: "))
start = time.perf_counter()
current_game = 0

open('datasets/features.bin', 'wb')
open('datasets/policies.txt', 'w')
open('datasets/values.txt', 'w')
for game in Pool().imap(generate_game, range(num_games)):
    for player_board, opponent_board, policy, value in game:
        with open('datasets/features.bin', 'ab') as f:
            # since the size of the group of symmetrical boards is 8
            # labels dealt with in the training loop to save disk space
            for i in range(2):
                for j in range(4):
                    position_string = format(player_board, '064b') + format(opponent_board, '064b')
                    byte_string = int(position_string, 2).to_bytes(16, byteorder='big')
                    f.write(byte_string)
                    player_board = rot_90_cw(player_board)
                    opponent_board = rot_90_cw(opponent_board)
                player_board = horizontal_mirror_image(player_board)
                opponent_board = horizontal_mirror_image(opponent_board)

        with open('datasets/policies.txt', 'a') as f:
            f.write(' '.join(map(str, policy)) + '\n')
        with open('datasets/values.txt', 'a') as f:
            f.write(str(value) + '\n')
    current_game += 1
    if current_game % 10 == 0:
        print("At Game #" + str(current_game))