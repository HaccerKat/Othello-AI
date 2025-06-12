
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import random

start = time.perf_counter()
for _ in range(10**4):
    black_board = 0
    white_board = 0
    for i in range(64):
        x = random.randint(0, 2)
        if x == 1:
            black_board |= (1 << i)
        elif x == 2:
            white_board |= (1 << i)
    player = random.randint(0, 1)
end = time.perf_counter()
print("Random Generation:", end - start)

# Get new boards for a given black and white board state test
# Unreasonable since random boards are not representative of actual game states, but useful for benchmarking.
# start = time.perf_counter()
# for _ in range(10**4):
#     black_board = 0
#     white_board = 0
#     for i in range(64):
#         x = random.randint(0, 2)
#         if x == 1:
#             black_board |= (1 << i)
#         elif x == 2:
#             white_board |= (1 << i)
#     player = random.randint(0, 1)
#     # print_both_boards(black_board, white_board)
#     # print("-----------------------------------")
#
#     boards = get_new_boards(black_board, white_board, player)
#     # boards_sanity = get_new_boards_sanity(black_board, white_board, player)
#     # if len(boards) != len(boards_sanity):
#     #     print("Boards length mismatch:", len(boards), len(boards_sanity))
#     #     print_both_boards(black_board, white_board)
#     #     print("-----------------------------------")
#     #     for b in boards:
#     #         b.__print__()
#     #     print("-----------------------------------")
#     #     for b in boards_sanity:
#     #         b.__print__()
#         # assert(False)
# end = time.perf_counter()
# print("Get New Boards:", end - start)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 65),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)

learning_rate = 0.3
batch_size = 64

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

model.train()

# Benchmarking the neural network evaluation
start = time.perf_counter()
for _ in range(10**5):
    with torch.no_grad():
        X = torch.rand((batch_size, 128), device=device)
        pred = model(X)
end = time.perf_counter()
print("NN Eval:", end - start)

model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

model_dynamic_quantized.eval()
start = time.perf_counter()
for _ in range(10**4):
    with torch.no_grad():
        X = torch.rand((batch_size, 128), device=device)
        pred = model_dynamic_quantized(X)
end = time.perf_counter()
print("NN Dynamic Quantized Eval:", end - start)

backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

# start = time.perf_counter()
# for _ in range(10**4):
#     with torch.no_grad():
#         X = torch.rand((batch_size, 128), device=device)
#         pred = model_static_quantized(X)
# end = time.perf_counter()
# print("NN Static Quantized Eval:", end - start)
# from multiprocessing import Pool
# import torch

# def nn_eval(input_tensor):
#     with torch.no_grad():
#         return model(input_tensor).numpy()

# pool = Pool(processes=4)  # Or whatever fits your CPU

# # Inside MCTS leaf expansion
# results = pool.map(nn_eval, list_of_input_tensors)
