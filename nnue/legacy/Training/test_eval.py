import torch
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

import torch

def load_model(model_class, checkpoint_path, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


model = load_model(NeuralNetwork, 'model_weights.pth')
# nnue_layer = [0] * 129  # 2 bits per square * 64 squares + 1 player bit
# grid = [['.' for _ in range(8)] for _ in range(8)]

# for i in range(8):
#     row = input().split()  # assuming space-separated input
#     for j in range(8):
#         grid[i][j] = row[j]
#         pos = 2 * (i * 8 + j)
#         if grid[i][j] == '0':
#             nnue_layer[pos + 1] = 1
#         if grid[i][j] == '1':
#             nnue_layer[pos] = 1

# player = int(input())
# nnue_layer[128] = player

# with torch.no_grad():
#     output = model(torch.tensor(nnue_layer, dtype=torch.float32))
#     print(output)

def decode_nnue_layer(nnue_layer):
    grid = [['.' for _ in range(8)] for _ in range(8)]

    for i in range(8):
        for j in range(8):
            pos = 2 * (i * 8 + j)
            bit1 = nnue_layer[pos + 1]
            bit0 = nnue_layer[pos]

            if bit1 == 1:
                grid[i][j] = '1'
            elif bit0 == 1:
                grid[i][j] = '0'
            else:
                grid[i][j] = '.'

    player = nnue_layer[128]
    return grid, player


filename = 'data3.bin'

with open(filename, 'rb') as f:
    raw = np.frombuffer(f.read(), dtype=np.uint8)

# Unpack all bits
all_bits = np.unpackbits(raw)

# Each sample = 136 bits (129 inputs + 5 unused + 2 label)
num_samples = all_bits.size // 136
all_bits = all_bits[:num_samples * 136]

# Reshape to (num_samples, 136)
data = all_bits.reshape((num_samples, 136))

# Slice out inputs and labels
inputs = data[:, :129]

labels = data[:, 134:]  # Single bit label per sample
labels = np.packbits(labels, axis = -1)
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
labels = labels / 128

x = 20
correct = 0
indices = torch.randperm(inputs.size(0))[:x]
sample_inputs = inputs[indices]
sample_labels = labels[indices]

with torch.no_grad():
    outputs = model(sample_inputs)

correct += (abs(sample_labels - outputs) < 0.5).type(torch.float).sum().item()
# Print results
for i in range(x):
    print(f"Sample {i + 1}:")
    print(f"Expected: {sample_labels[i].cpu().numpy()}")
    print(f"Predicted: {outputs[i].cpu().numpy()}")
    grid, player = decode_nnue_layer(sample_inputs[i].cpu().numpy())
    print(f"Player: {player}")
    for row in grid:
        print(" ".join(row))
    print("----------------------------------")

correct /= x
print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")