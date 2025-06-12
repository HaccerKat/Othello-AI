import random
from board import Board
from mcts import mcts
import numpy as np

def simulate_game(parameters):
    identifier, control_model, experimental_model, num_simulations, exploration_constant = parameters
    control_player = random.randint(0, 1)
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000
    board = Board(initial_player_board, initial_opponent_board, current_player)
    sum_full_policy = np.zeros(64)
    sum_legal_moves = np.zeros(64)
    while not board.game_ends():
        if current_player == control_player:
            (player_board, opponent_board) = mcts(board, control_model, False, True, num_simulations, exploration_constant)
        else:
            (player_board, opponent_board) = mcts(board, experimental_model, False, True, num_simulations, exploration_constant)
            sum_full_policy += board.get_full_policy()
            sum_legal_moves += board.legal_moves
        current_player = 1 - current_player
        board = Board(player_board, opponent_board, current_player)

    winner = board.get_winner()
    current_is_control = current_player == control_player
    if winner == 0:
        # draws += 1
        return 0, sum_full_policy, sum_legal_moves
    elif (current_is_control and winner == 1) or (not current_is_control and winner == -1):
        # control_wins += 1
        return -1, sum_full_policy, sum_legal_moves
    else:
        # experimental_wins += 1
        return 1, sum_full_policy, sum_legal_moves