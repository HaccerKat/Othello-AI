import board_helper as bh
import math
import numpy as np
import random
import time
import globals
import torch
class Board:
    def __init__(self, player_board, opponent_board, player, parent = None):
        self.player_board = player_board
        self.opponent_board = opponent_board
        self.player = player
        self.parent = parent
        self.current_child = 0
        self.next_boards = []
        self.value_head = 0
        # For MCTS:
        self.visited_count = 0 # N(s) and N(s, a) in PUCT
        self.sum_eval = 0 # Divide by visited_count to get Q(s, a) in PUCT
        self.mcts_policy = []
        self.full_policy = np.zeros(64)
        self.legal_moves = np.zeros(64)

    def print(self):
        print("Board State:")
        bh.print_both_boards(self.player_board, self.opponent_board)
        print(f"Player: {self.player}")
        print(f"Value Head: {self.value_head}")
        print(f"Visited Count: {self.visited_count}")
        print(f"Sum Eval: {self.sum_eval}")
        print(f"Current Child: {self.current_child}")
        print(f"MCTS Policy: {self.mcts_policy}")
        test = []
        test2 = []
        for policy_value, move, child in self.next_boards:
            test.append((policy_value, move))
            test2.append(child.sum_eval / child.visited_count)
        print(f"NN Policy: {test}")
        print(f"NN Value Head: {test2}")
        self.get_full_policy()
        print(f"Full Policy: {self.full_policy}")
        print("---------------------------------------------")

    def find_next_boards(self, model):
        tensor = bh.to_tensor(self.player_board, self.opponent_board)

        policy, self.value_head = model(tensor)
        policy = torch.nn.functional.softmax(policy, dim=0)
        policy = policy.detach().numpy()

        self.value_head = self.value_head.item()
        empty_board = ~(self.player_board | self.opponent_board) & 0xFFFFFFFFFFFFFFFF
        legal = bh.find_legal_moves(self.player_board, self.opponent_board)
        for i in range(8):
            for j in range(8):
                if not (legal >> (i * 8 + j) & 1):
                    continue
                flip = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        flip_line = 0
                        x, y = i + di, j + dj
                        while 0 <= x < 8 and 0 <= y < 8:
                            index = x * 8 + y
                            if empty_board >> index & 1:
                                break
                            if self.opponent_board >> index & 1:
                                flip_line |= (1 << index)
                            elif self.player_board >> index & 1:
                                break
                            x += di
                            y += dj
                        if 0 <= x < 8 and 0 <= y < 8 and (self.player_board >> (x * 8 + y) & 1):
                            flip |= flip_line
                if flip:
                    index = i * 8 + j
                    new_player_board = self.player_board ^ (1 << index) ^ flip
                    new_opponent_board = self.opponent_board ^ flip
                    self.next_boards.append((policy[index], index, Board(new_opponent_board, new_player_board, 1 - self.player, self)))

        if not self.next_boards:
            self.next_boards.append((1, -1, Board(self.opponent_board, self.player_board, 1 - self.player, self)))

        sorted(self.next_boards, reverse=True)
    
    def game_ends(self):
        legal = bh.find_legal_moves(self.player_board, self.opponent_board)
        legal_opponent = bh.find_legal_moves(self.opponent_board, self.player_board)
        return legal == 0 and legal_opponent == 0

    def get_winner(self):
        player_count = bh.get_points(self.player_board)
        opponent_count = bh.get_points(self.opponent_board)
        if player_count > opponent_count:
            return 1
        elif opponent_count > player_count:
            return -1
        else:
            return 0

    # MCTS Functions
    def expand(self, model):
        if not self.next_boards:
            self.find_next_boards(model)

        assert len(self.next_boards) > 0
        if self.current_child >= len(self.next_boards):
            raise IndexError("No more children to expand.")
        new_board = self.next_boards[self.current_child][2]
        self.current_child += 1
        return new_board
    
    def select(self, exploration_constant):
        # PUCT selection
        max_puct = -1.0
        sum_policy_value = 0.0
        new_board = None
        for policy_value, move, child in self.next_boards:
            sum_policy_value += policy_value
        for policy_value, move, child in self.next_boards:
            # -child.sum_eval since we choose the move that makes it the worst for the opponent (player board and opponent board are swapped every move)
            x = (-child.sum_eval / child.visited_count) + exploration_constant * policy_value / sum_policy_value * math.sqrt(self.visited_count) / (1 + child.visited_count)
            if x >= max_puct:
                max_puct = x
                new_board = child
        return new_board

    def backpropagate(self, model):
        if not self.next_boards:
            self.find_next_boards(model)

        start = time.perf_counter()
        if self.game_ends():
            reward = self.get_winner()
        else:
            reward = self.value_head
        node = self
        while node is not None:
            node.sum_eval += reward
            node.visited_count += 1
            node = node.parent
            reward *= -1
        end = time.perf_counter()
        globals.state['time_eval'] += end - start

    def get_next_board(self, gameplay = False):
        for policy_value, move, child in self.next_boards:
            # new policy generated by MCTS visit count probabilities
            self.mcts_policy.append(child.visited_count / self.visited_count)

        # selects top move, no randomness except early moves to prevent only 2 possible game lines from being played
        if gameplay:
            if bh.get_points(self.player_board) + bh.get_points(self.opponent_board) < 10:
                selection = random.choices(self.next_boards, k=1)[0]
            else:
                index = np.argmax(self.mcts_policy)
                selection = self.next_boards[index]
            return selection[2].player_board, selection[2].opponent_board

        alpha = min(1.0, 10 / len(self.mcts_policy))
        epsilon = 0.25
        mcts_policy = (1 - epsilon) * np.array(self.mcts_policy) + epsilon * np.random.dirichlet(np.ones(len(self.mcts_policy)) * alpha)
        # random selection weighted using policy + dirichlet distribution
        # using lists are faster than numpy arrays when doing random selection only once
        selection = random.choices(self.next_boards, weights=mcts_policy, k=1)[0]
        if bh.get_points(self.player_board) + bh.get_points(self.opponent_board) < 10:
            selection = random.choices(self.next_boards, k=1)[0]
        # child board states of the selection
        return selection[2].player_board, selection[2].opponent_board

    def get_full_policy(self):
        index = 0
        sum_values = 0
        for policy_value, move, child in self.next_boards:
            self.full_policy[move] += self.mcts_policy[index]
            self.legal_moves[move] = 1
            sum_values += self.full_policy[move]
            index += 1
        if sum_values > 0:
            self.full_policy /= sum_values
        else:
            # indicates a skip turn and not to add in dataset
            self.full_policy[0] = -1
        return self.full_policy

