import torch
from torch.utils.data import DataLoader
from nn import load_model
import numpy as np
from multiprocessing import Pool
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import entropy
import time
from torch.utils.data import ConcatDataset

from initialization import NeuralNetwork, Dataset
import board_helper as bh
from generate_game_helper import generate_game
from training_helper import train_loop, test_loop
from simulate_game_helper import simulate_game

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 3e-3
BUFFER_START = 5
# MCTS hyperparameters
num_simulations = 100
exploration_constant = math.sqrt(2)

buffer = None
def training_loop(generation, model):
    global buffer
    inputs_list, policies_list, values_list = [], [], []
    current_game = 0
    num_games = 288
    num_positions = 0

    sum_entropy = 0

    jobs = [(i, model, model, num_simulations, exploration_constant) for i in range(num_games)]
    for game in Pool().imap(generate_game, jobs):
        for player_board, opponent_board, policy, value in game:
            # since the size of the group of symmetrical boards is 8
            sum_entropy += entropy(policy)
            num_positions += 1
            policy = torch.from_numpy(policy)
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
        if current_game % 10 == 0:
            print("Generating Game #" + str(current_game))

    avg_entropy = sum_entropy / num_positions

    inputs = torch.tensor(inputs_list).to(device)
    inputs = torch.reshape(inputs, (-1, 2, 8, 8))
    policies = torch.tensor(policies_list).to(device)
    values = torch.tensor(values_list).to(device)
    dataset = Dataset(inputs, policies, values)
    training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if buffer is not None:
        # eventually converges to len(buffer_dataset) = len(dataset)
        buffer_dataset, buffer = torch.utils.data.random_split(buffer, [0.4, 0.6])
        training_data = ConcatDataset([training_data, buffer_dataset])

    BATCH_SIZE = 64
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    best_test_loss = float("inf")
    patience = 2
    patience_counter = 0
    epochs = 7
    best_nn = None
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer, BATCH_SIZE)
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
    return best_nn, best_policy_loss, best_value_loss, best_test_loss, patience == patience_counter, avg_entropy

PLOT_MODULO = 5
def update_elo(generation):
    control_model = load_model(NeuralNetwork, 'models/model_weights_' + str(generation - PLOT_MODULO) + '.pth')
    experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + str(generation) + '.pth')
    current_game = 0

    sum_full_policy = np.zeros(64)
    sum_legal_moves = np.zeros(64)
    num_games = 100
    draws, control_wins, experimental_wins = 0, 0, 0
    jobs = [(i, control_model, experimental_model, num_simulations, exploration_constant) for i in range(num_games)]
    for result, full_policy, legal_moves in Pool().imap(simulate_game, jobs):
        if result == 0:
            draws += 1
        elif result == -1:
            control_wins += 1
        else:
            experimental_wins += 1
        current_game += 1
        sum_full_policy += full_policy
        sum_legal_moves += legal_moves
        if current_game % 10 == 0:
            print("Simulating Game #" + str(current_game))

    num_games -= draws
    experimental_wr = experimental_wins / num_games
    probability = 1.0 * experimental_wins / num_games
    error = 1.96 * math.sqrt(probability * (1 - probability) / num_games)
    lower_bound = 100.0 * (probability - error)
    upper_bound = 100.0 * (probability + error)
    print("Number of Draws:", draws)
    print("Number of Control Wins:", control_wins)
    print("Number of Experimental Wins:", experimental_wins)
    print("Experimental WR: " + str(experimental_wr * 100.0) + "%")
    print("95% Confidence Interval: [" + str(lower_bound) + "%, " + str(upper_bound) + "%]")

    avg_full_policy = sum_full_policy / (sum_legal_moves + 1)
    avg_full_policy = (avg_full_policy ** 2) * 10
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

start = time.perf_counter()
generation = 0
elo = 0

patience_exceeded_counter = 0
patience_exceeded_counter_limit = 5
y_elo_list = [[0]]
x_elo_list = [0]
# total loss, policy loss, value loss
y_validation_loss_list = [[], [], []]
print(y_validation_loss_list)
y_entropy_list = [[]]
x_general_list = []

model = NeuralNetwork().to(device)
state_dict = model.state_dict()
torch.save(model.state_dict(), './models/model_weights_0.pth')

# simulate 100 games every 10 generations for testing strength
while True:
    print('---------------------------------' + "Training Generation " + str(generation + 1) + '---------------------------------')
    bestNN, policy_loss, value_loss, test_loss, patience_exceeded, avg_entropy = training_loop(generation, model)
    patience_exceeded_counter += patience_exceeded
    if patience_exceeded_counter >= patience_exceeded_counter_limit:
        print('---------------------------------' + "Patience Exceeded Counter Exceeded" + '---------------------------------')
        learning_rate /= 2
        if num_simulations <= 1000:
            num_simulations *= 8
            num_simulations //= 5
        print("New Learning Rate: " + str(learning_rate))
        print("New MCTS Number of Simulations: " + str(num_simulations))
        patience_exceeded_counter = 0

    generation += 1
    x_general_list.append(generation)
    y_validation_loss_list[0].append(test_loss)
    y_validation_loss_list[1].append(policy_loss)
    y_validation_loss_list[2].append(value_loss)
    y_entropy_list[0].append(avg_entropy)
    if generation % PLOT_MODULO == 0 and generation > 0:
        elo += update_elo(generation)
        x_elo_list.append(generation)
        y_elo_list[0].append(elo)
        plot(x_elo_list, y_elo_list, ('Elo',), generation, 'Elo', 'elo')
        plot(x_general_list, y_validation_loss_list, ('Loss', 'Policy Loss', 'Value Loss'), generation, 'Validation Loss', 'validation_loss')
        plot(x_general_list, y_entropy_list, ('Entropy',), generation, 'Entropy', 'entropy')
    
    end = time.perf_counter()
    duration = end - start
    if duration >= 86400:
        break