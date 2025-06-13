import random
from board import Board
from mcts import mcts
def generate_game(parameters):
    identifier, control_model, experimental_model, num_simulations, exploration_constant = parameters
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
            (player_board, opponent_board) = mcts(board, control_model, False, False, num_simulations, exploration_constant)
        else:
            (player_board, opponent_board) = mcts(board, experimental_model, False, False, num_simulations, exploration_constant)

        current_player = 1 - current_player
        board = Board(player_board, opponent_board, current_player)
        print(p)

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
        value_mcts = position.sum_eval / position.visited_count
        return_game.append((position.player_board, position.opponent_board, policy, value_mcts))
        return_game.append((position.player_board, position.opponent_board, policy, winner))

    return return_game