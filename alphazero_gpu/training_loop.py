import torch
from torch.utils.data import DataLoader
from nn_init import load_model
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import entropy
import time
from torch.utils.data import ConcatDataset
import torch.multiprocessing as mp
import os
import json

from nn_init import NeuralNetwork, Dataset
import board_helper as bh
from generate_games import generate_games
from training_helper import train_loop, test_loop
from simulate_games import simulate_game
from multiprocessing_helper import execute_mp, execute_gpu

learning_rate = 0.002
BUFFER_START = 3
# MCTS hyperparameters
num_simulations = 100
exploration_constant_training = 4
exploration_constant_testing = math.sqrt(2)

buffer = None
def training_loop(generation, model, device):
    global buffer
    inputs_list, policies_list, values_list = [], [], []
    current_game = 0
    num_games_to_simulate = 512
    inference_batch_size = 64
    # guarantees equal number of games between control and experimental starting first
    assert num_games_to_simulate % (inference_batch_size * 2) == 0
    num_positions = 0

    sum_entropy = 0

    model.eval()
    model.share_memory()
    jobs = [(i, model, model, num_simulations, inference_batch_size, exploration_constant_training, 0.5) for i in range(num_games_to_simulate // inference_batch_size)]

    print("Generating Games...")
    if device == "cpu":
        # games = execute_mp(generate_game, jobs)
        assert False
    else:
        batches = execute_gpu(generate_games, jobs)

    total_position_importance = np.zeros(64)
    total_cnt = np.zeros(64)
    for batch, position_importance, cnt in batches:
        for player_board, opponent_board, policy, value in batch:
            # since the size of the group of symmetrical boards is 8
            sum_entropy += entropy(policy)
            num_positions += 1
            policy = torch.from_numpy(policy)
            # position_string = format(player_board, '064b') + format(opponent_board, '064b')
            # appends position_string as int array of 0s and 1s
            # inputs_list.append(list(map(int, list(position_string))))
            # policies_list.append(policy.tolist())
            # values_list.append([value])
            for i in range(2):
                for j in range(4):
                    position_string = format(player_board, '064b') + format(opponent_board, '064b')
                    # appends position_string as int array of 0s and 1s
                    inputs_list.append(list(map(int, list(position_string))))
                    policies_list.append(policy.tolist())
                    values_list.append([value])
                    player_board = bh.rot_90_cw(player_board)
                    opponent_board = bh.rot_90_cw(opponent_board)
                    policy = bh.rot_90_cw_policy(policy)
                player_board = bh.horizontal_mirror_image(player_board)
                opponent_board = bh.horizontal_mirror_image(opponent_board)
                policy = bh.horizontal_mirror_image_policy(policy)

        current_game += 1
        total_position_importance += position_importance
        total_cnt += cnt

    avg_entropy = sum_entropy / num_positions

    total_position_importance /= total_cnt
    plt.imshow(total_position_importance.reshape(8, 8), cmap='Reds')
    plt.colorbar()
    plt.title("Location of Crucial Moves")
    plt.show()

    inputs = torch.tensor(inputs_list)
    inputs = torch.reshape(inputs, (-1, 2, 8, 8))
    policies = torch.tensor(policies_list)
    values = torch.tensor(values_list)

    dataset = Dataset(inputs, policies, values)
    training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if buffer is not None:
        buffer_dataset, buffer = torch.utils.data.random_split(buffer, [0.5, 0.5])
        # discard faster in earlier generations
        x = max(0.2, 1.0 - 0.03 * generation)
        discard_dataset, keep_dataset = torch.utils.data.random_split(buffer_dataset, [x, 1 - x])
        training_data = ConcatDataset([training_data, buffer_dataset])
        buffer = ConcatDataset([buffer, keep_dataset])
        del discard_dataset

    BATCH_SIZE = 128
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    best_test_loss = float("inf")
    patience = 1
    patience_counter = 0
    epochs = 2

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=epochs)

    best_nn = None
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, scheduler, BATCH_SIZE)
        policy_loss, value_loss, test_loss = test_loop(test_dataloader, model)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_policy_loss = policy_loss
            best_value_loss = value_loss
            best_nn = model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # assign to buffer
    if generation < BUFFER_START:
        pass
    elif buffer is None:
        buffer = dataset
    else:
        buffer = ConcatDataset([buffer, dataset])
    torch.save(best_nn.state_dict(), 'models/model_weights_' + str(int(generation) + 1) + '.pth')
    torch.cuda.empty_cache()
    return best_nn, best_policy_loss, best_value_loss, best_test_loss, patience == patience_counter, avg_entropy

plot_modulo = 1
def update_elo(generation, device):
    control_model = load_model(NeuralNetwork, 'models/model_weights_' + str(generation - plot_modulo) + '.pth')
    experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + str(generation) + '.pth')
    current_game = 0

    control_model.eval()
    control_model.share_memory()
    experimental_model.eval()
    experimental_model.share_memory()

    sum_full_policy = np.zeros(64)
    sum_legal_moves = np.zeros(64)
    num_games_to_simulate = 256
    inference_batch_size = 32
    draws, control_wins, experimental_wins = 0, 0, 0
    jobs = [(i, control_model, experimental_model, num_simulations, inference_batch_size, exploration_constant_testing) for i in range(num_games_to_simulate // inference_batch_size)]

    print("Simulating Games...")
    if device == "cpu":
        # games = execute_mp(simulate_game, jobs)
        pass
    else:
        games = execute_gpu(simulate_game, jobs)

    for result_draws, result_control_wins, result_experimental_wins, full_policy, legal_moves in games:
        draws += result_draws
        control_wins += result_control_wins
        experimental_wins += result_experimental_wins
        current_game += 1
        sum_full_policy += full_policy
        sum_legal_moves += legal_moves

    num_games_to_simulate -= draws
    experimental_wr = experimental_wins / num_games_to_simulate
    probability = 1.0 * experimental_wins / num_games_to_simulate
    error = 1.96 * math.sqrt(probability * (1 - probability) / num_games_to_simulate)
    lower_bound = 100.0 * (probability - error)
    upper_bound = 100.0 * (probability + error)
    print("Number of Draws:", draws)
    print("Number of Control Wins:", control_wins)
    print("Number of Experimental Wins:", experimental_wins)
    print("Experimental WR: " + str(experimental_wr * 100.0) + "%")
    print("95% Confidence Interval: [" + str(lower_bound) + "%, " + str(upper_bound) + "%]")

    avg_full_policy = sum_full_policy / (sum_legal_moves + 0.0001)
    plt.imshow(avg_full_policy.reshape(8, 8), cmap='Reds')
    plt.colorbar()
    plt.title("Location of Crucial Moves")
    plt.savefig('plots/crucial_moves/generation_' + str(generation) + '.png')
    plt.clf()
    plt.close()

    elo_gain = 400 * math.log10(1 / (1 - experimental_wr) - 1)
    return elo_gain

# def plot_smooth(x, y, generation, name):
#     xnew = np.linspace(min(x), max(x), 300)
#     spl = make_interp_spline(x, y, k=3)
#     power_smooth = spl(xnew)
#     plt.plot(xnew, power_smooth)
#     if name == 'elo':
#         plt.xlabel('Generation')
#         plt.ylabel('Elo')
#         plt.title('Elo vs. Generation')
#     else:
#         plt.xlabel('Generation')
#         plt.ylabel('Validation Loss')
#         plt.title('Validation Loss vs. Generation')
#     plt.savefig('plots/' + name + '/generation_' + str(generation) + '.png')
#     plt.clf()
#     plt.close()

def plot(x, y, labels, generation, title, folder_name):
    for i in range(len(y)):
        plt.plot(x, y[i], label=labels[i])
    plt.xlabel('Generation')
    plt.ylabel(title)
    plt.title(title + ' vs. Generation')
    plt.savefig('plots/' + folder_name + '/generation_' + str(generation) + '.png')
    plt.clf()
    plt.close()

def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    torch.set_num_threads(1)
    try:
        with open('reload_model_data.json', 'r') as f:
            json_data = json.load(f)
            generation = json_data['generation']
            elo = json_data['elo']
            patience_exceeded_counter = json_data['patience_exceeded_counter']
            y_elo_list = json_data['y_elo_list']
            x_elo_list = json_data['x_elo_list']
            y_validation_loss_list = json_data['y_validation_loss_list']
            y_entropy_list = json_data['y_entropy_list']
            x_general_list = json_data['x_general_list']
            model = load_model(NeuralNetwork, 'models/model_weights_' + str(generation) + '.pth')
        global buffer
        buffer = torch.load('reload_tensor.pth')
    except (FileNotFoundError, json.JSONDecodeError):
        generation = 0
        elo = 0
        patience_exceeded_counter = 0
        y_elo_list = [[0]]
        x_elo_list = [0]
        y_validation_loss_list = [[], [], []]
        y_entropy_list = [[]]
        x_general_list = []
        model = NeuralNetwork().to(device)
        torch.save(model.state_dict(), './models/model_weights_0.pth')

    PATIENCE_EXCEEDED_COUNTER_LIMIT = 5
    while True:
        start = time.perf_counter()
        print('---------------------------------' + "Training Generation " + str(generation + 1) + '---------------------------------')
        global learning_rate
        global num_simulations
        global plot_modulo
        if generation >= 6:
            learning_rate = 0.0005
            num_simulations = 200
            plot_modulo = 3
        if generation >= 15:
            learning_rate = 0.0002
            num_simulations = 400
            plot_modulo = 3
        if generation >= 25:
            learning_rate = 0.0001
            num_simulations = 800
            plot_modulo = 3
        bestNN, policy_loss, value_loss, test_loss, patience_exceeded, avg_entropy = training_loop(generation, model, device)
        model = bestNN
        patience_exceeded_counter += patience_exceeded
        if patience_exceeded_counter >= PATIENCE_EXCEEDED_COUNTER_LIMIT:
            print('---------------------------------' + "Patience Exceeded Counter Exceeded" + '---------------------------------')
            patience_exceeded_counter = 0

        generation += 1
        x_general_list.append(generation)
        y_validation_loss_list[0].append(test_loss)
        y_validation_loss_list[1].append(policy_loss)
        y_validation_loss_list[2].append(value_loss)
        y_entropy_list[0].append(avg_entropy)
        if generation % plot_modulo == 0 and generation > 0:
            elo_gain = update_elo(generation, device)
            elo += elo_gain
            x_elo_list.append(generation)
            y_elo_list[0].append(elo)
            plot(x_elo_list, y_elo_list, ('Elo',), generation, 'Elo', 'elo')
            plot(x_general_list, y_validation_loss_list, ('Loss', 'Policy Loss', 'Value Loss'), generation, 'Validation Loss', 'validation_loss')
            plot(x_general_list, y_entropy_list, ('Entropy',), generation, 'Entropy', 'entropy')
        
        end = time.perf_counter()

        # In case of a crash/Azure ML cuts off low priority cluster
        json_data = {
            "generation": generation,
            "elo": elo,
            "patience_exceeded_counter": patience_exceeded_counter,
            "y_elo_list": y_elo_list,
            "x_elo_list": x_elo_list,
            "y_validation_loss_list": y_validation_loss_list,
            "y_entropy_list": y_entropy_list,
            "x_general_list": x_general_list,
        }
        
        with open('reload_model_data.json', 'w') as f:
            json.dump(json_data, f, indent = 4)

        global buffer
        torch.save(buffer, 'reload_tensor.pth')
        duration = end - start
        print(f"Generation {generation} completed in {duration:.2f} seconds")
        print(torch.cuda.memory_allocated() / 1e6, "MB")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()