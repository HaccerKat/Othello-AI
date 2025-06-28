import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import os

from mcts import mcts_mp
from board import Board
import board_helper as bh
from nn_init import NeuralNetwork, load_model
from multiprocessing_helper import execute_mp

def simulate_game(parameters):
    identifier, control_model, experimental_model, num_simulations, num_games, exploration_constant = parameters
    control_player = random.randint(0, 1)
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000
    boards = [Board(initial_player_board, initial_opponent_board, current_player) for _ in range(num_games)]
    sum_full_policy = np.zeros(64)
    sum_legal_moves = np.zeros(64)
    draws, control_wins, experimental_wins = 0, 0, 0
    move_num = 0
    while boards:
        if current_player == control_player:
            new_boards = mcts_mp(boards, control_model, len(boards), False, True, num_simulations, exploration_constant)
        else:
            new_boards = mcts_mp(boards, experimental_model, len(boards), False, True, num_simulations, exploration_constant)
            for board in boards:
                sum_full_policy += board.get_full_policy()
                sum_legal_moves += board.legal_moves

        # print("Sum full policy:", sum_full_policy)
        # print("Sum legal moves:", sum_legal_moves)
        current_player = 1 - current_player
        boards = []
        for player_board, opponent_board in new_boards:
            board = Board(player_board, opponent_board, current_player)
            # bh.print_both_boards(player_board, opponent_board)
            current_is_control = current_player == control_player
            if board.game_ends():
                winner = board.get_winner()
                if winner == 0:
                    draws += 1
                elif (current_is_control and winner == 1) or (not current_is_control and winner == -1):
                    control_wins += 1
                else:
                    experimental_wins += 1
            else:
                boards.append(board)

        move_num += 1
        print("At move number:", move_num)
    return draws, control_wins, experimental_wins, sum_full_policy, sum_legal_moves

# 0 - Black
# 1 - White
def main():
    pass
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    #
    # torch.set_num_threads(1)


    # Fix later
    # nn_name_control = input("Enter the control's model name: ")
    # nn_name_experimental = input("Enter the experimental's model name: ")
    # control_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_control + '.pth')
    # experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_experimental + '.pth')
    # control_model.eval()
    # experimental_model.eval()
    #
    # draws, control_wins, experimental_wins = 0, 0, 0
    #
    # num_games = int(input("Enter the number of games: "))
    # start = time.perf_counter()
    # current_game = 0
    #
    # sum_full_policy = np.zeros(64)
    # sum_legal_moves = np.zeros(64)
    # control_model.eval()
    # control_model.share_memory()
    # experimental_model.eval()
    # experimental_model.share_memory()
    # jobs = [(i, control_model, experimental_model, 100, math.sqrt(2)) for i in range(num_games)]
    #
    # print("Generating Games...")
    # games = execute_mp(simulate_game, jobs)
    # for result_draws, result_control_wins, result_experimental_wins, full_policy, legal_moves in games:
    #     draws += result_draws
    #     control_wins += result_control_wins
    #     experimental_wins += result_experimental_wins
    #     current_game += 1
    #     sum_full_policy += full_policy
    #     sum_legal_moves += legal_moves
    #     if current_game % 10 == 0:
    #         print("At Game #" + str(current_game))
    #
    # print(sum_full_policy)
    # print(sum_legal_moves)
    # num_games -= draws
    # experimental_WR = experimental_wins / num_games
    # probability = 1.0 * experimental_wins / num_games
    # error = 1.96 * math.sqrt(probability * (1 - probability) / num_games)
    # lower_bound = 100.0 * (probability - error)
    # upper_bound = 100.0 * (probability + error)
    # print("Number of Draws:", draws)
    # print("Number of Control Wins:", control_wins)
    # print("Number of Experimental Wins:", experimental_wins)
    # print("Experimental WR: " + str(experimental_WR * 100.0) + "%")
    # print("95% Confidence Interval: [" + str(lower_bound) + "%, " + str(upper_bound) + "%]")
    #
    # avg_full_policy = sum_full_policy / (sum_legal_moves + 1)
    # avg_full_policy = (avg_full_policy ** 2) * 10
    # print("Control Avg Full Policy:")
    # plt.imshow(avg_full_policy.reshape(8, 8), cmap='Reds')
    # plt.colorbar()
    # plt.show()
    #
    # end = time.perf_counter()
    # print("Execution time (s):", end - start)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()