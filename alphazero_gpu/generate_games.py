import random
from nn_init import NeuralNetwork, load_model
from mcts import mcts_mp
from board import Board
from board_helper import horizontal_mirror_image, rot_90_cw, get_points
import time
import os
import globals
import numpy as np
import math
from multiprocessing_helper import execute_gpu
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

# def generate_game(parameters):
#     identifier, control_model, experimental_model, num_simulations, exploration_constant = parameters
#     print("Game Gen " + str(identifier) + " is starting")
#     control_player = random.randint(0, 1)
#     current_player = 0
#     initial_player_board = 0x0000000810000000
#     initial_opponent_board = 0x0000001008000000
#     if control_player == 1:
#         initial_player_board = 0x0000000810000000
#         initial_opponent_board = 0x00000001008000000
#     board = Board(initial_player_board, initial_opponent_board, current_player)
#     game = []
#     # prevent early game positions from flooding the dataset
#     p = 3125
#     while not board.game_ends():
#         start = time.perf_counter()
#         if random.randint(0, p) == p:
#             game.append(board)
#
#         p //= 5
#         if current_player == control_player:
#             (player_board, opponent_board) = mcts(board, control_model, False, False, num_simulations, exploration_constant)
#         else:
#             (player_board, opponent_board) = mcts(board, experimental_model, False, False, num_simulations, exploration_constant)
#
#         current_player = 1 - current_player
#         board = Board(player_board, opponent_board, current_player)
#         print("Game Gen " + str(identifier) + " has successfully finished making a move")
#         end = time.perf_counter()
#         globals.state['time_eval_2'] += end - start
#         # print("NN Inference Time Only:", globals.state['time_eval'])
#         # print("Full Game Generation Time:", globals.state['time_eval_2'])
#
#     # end board not included
#     # game.append(board)
#     game_winner = board.get_winner()
#     return_game = []
#     # weight full games more in later generations (make epsilon closer to 1)
#     epsilon = 0.5
#     for position in game:
#         policy = position.get_full_policy()
#         if policy[0] == -1:
#             # indicates a skip turn and not to add in dataset
#             continue
#         winner = game_winner * (1 if position.player == current_player else -1)
#         value_mcts = position.sum_eval / position.visited_count
#         # return_game.append((position.player_board, position.opponent_board, policy, value_mcts))
#         # return_game.append((position.player_board, position.opponent_board, policy, winner))
#         return_game.append((position.player_board, position.opponent_board, policy, (value_mcts + winner) / 2))
#     return return_game

# multiprocessing simulating games
def generate_games(parameters):
    identifier, control_model, experimental_model, num_simulations, num_games, exploration_constant, epsilon = parameters
    print("Game Gen " + str(identifier) + " is starting")
    # half-half on whether control or experimental goes first
    control_player = identifier % 2
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000

    # each game is given an identifier at the start
    boards_and_identifier = [(i, Board(initial_player_board, initial_opponent_board, current_player)) for i in range(num_games)]
    games = [[] for _ in range(num_games)]
    dataset = []
    move_num = 0
    while boards_and_identifier:
        start = time.perf_counter()
        boards_only = []
        for i, board in boards_and_identifier:
            boards_only.append(board)

        exploration_constant_multiplier = 0.8 + move_num * 0.03
        if move_num >= 30:
            exploration_constant_multiplier = max(0.8, 1.7 - (move_num - 30) * 0.06)
        if current_player == control_player:
            new_boards = mcts_mp(boards_only, control_model, len(boards_only), False, False, num_simulations, exploration_constant * exploration_constant_multiplier)
        else:
            new_boards = mcts_mp(boards_only, experimental_model, len(boards_only), False, False, num_simulations, exploration_constant * exploration_constant_multiplier)

        for i, board in boards_and_identifier:
            games[i].append((board.player, board.player_board, board.opponent_board, board.get_full_policy(), board.sum_eval / board.visited_count))

        current_player = 1 - current_player
        tmp_boards_and_identifier = []
        for i, (player_board, opponent_board) in enumerate(new_boards):
            identifier = boards_and_identifier[i][0]
            new_board = Board(player_board, opponent_board, current_player)
            if new_board.game_ends():
                game_winner = new_board.get_winner()
                for player, player_board, opponent_board, policy, value_mcts in games[identifier]:
                    if policy[0] == -1:
                        # indicates a skip turn and not to add in dataset
                        continue
                    winner = game_winner * (1 if player == current_player else -1)
                    if epsilon == 0.0 and get_points(player_board) + get_points(opponent_board) > 50:
                        # used in nnue distillation
                        # this is done to reduce noise by cutting highly tactical (and noisy) positions
                        pass
                    else:
                        dataset.append((player_board, opponent_board, policy, (1 - epsilon) * value_mcts + epsilon * winner))
            else:
                tmp_boards_and_identifier.append((identifier, new_board))

        boards_and_identifier = tmp_boards_and_identifier
        end = time.perf_counter()
        globals.state['time_eval_2'] += end - start
        move_num += 1
        print("At move number:", move_num)
        # print("NN Inference Time Only:", globals.state['time_eval'])
        # print("Full Game Generation Time:", globals.state['time_eval_2'])
        # print("Just Backpropagation Time:", globals.state['time_eval_3'])

    position_importance = np.zeros(64)
    cnt = np.zeros(64)
    for player_board, opponent_board, policy, value in dataset:
        for i in range(64):
            if (player_board >> i) & 1:
                position_importance[i] += value
                cnt[i] += 1
            if (opponent_board >> i) & 1:
                position_importance[i] -= value
                cnt[i] += 1
    return dataset, position_importance, cnt

def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.set_num_threads(1)

    model_numbers = [54, 63, 66]
    # open('datasets/features.bin', 'wb')
    # open('datasets/values.txt', 'w')
    while True:
        m1 = model_numbers[random.randint(0, len(model_numbers) - 1)]
        m2 = model_numbers[random.randint(0, len(model_numbers) - 1)]
        print("Control Model Number: " + str(m1))
        print("Experimental Model Number: " + str(m2))
        control_model = load_model(NeuralNetwork, 'models/model_weights_' + str(m1) + '.pth')
        experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + str(m2) + '.pth')
        control_model.eval()
        experimental_model.eval()
        control_model.share_memory()
        experimental_model.share_memory()
        num_games_to_generate = 2048
        start = time.perf_counter()
        inference_batch_size = 256
        num_simulations = 800
        exploration_constant = 0.8
        jobs = [(i, control_model, experimental_model, num_simulations, inference_batch_size, exploration_constant, 0.0) for
                i in range(num_games_to_generate // inference_batch_size)]

        print("Generating Games...")
        games = execute_gpu(generate_games, jobs)
        for dataset, position_importance, cnt in games:
            for player_board, opponent_board, policy, value in dataset:
                with open('datasets/features.bin', 'ab') as f, open('datasets/values.txt', 'a') as g:
                    for _ in range(2): # swap player and opponent's pieces
                        for i in range(2): # flip horizontally
                            for j in range(4): # rotate 90 degrees
                                position_string = format(player_board, '064b') + format(opponent_board, '064b')
                                byte_string = int(position_string, 2).to_bytes(16, byteorder='big')
                                f.write(byte_string)
                                g.write(str(value) + ' ')
                                player_board = rot_90_cw(player_board)
                                opponent_board = rot_90_cw(opponent_board)
                            player_board = horizontal_mirror_image(player_board)
                            opponent_board = horizontal_mirror_image(opponent_board)
                        value *= -1
                        player_board, opponent_board = opponent_board, player_board

        end = time.perf_counter()
        print("Execution time (s):", end - start)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
