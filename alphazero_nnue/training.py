import numpy as np
import torch
from torch.utils.data import DataLoader
from board_helper import horizontal_mirror_image_policy, rot_90_cw_policy
def load_128bit_samples(filename):
    with open(filename, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    all_bits = np.unpackbits(raw)
    # Each sample = 128 bits (128 inputs)
    num_samples = all_bits.size // 128

    inputs = all_bits.reshape((num_samples, 128))
    return inputs

def load_policy(filename, n):
    policy = torch.empty(n * 8, 64, dtype=torch.float32)
    with open(filename, 'r') as f:
        for i in range(n):
            base = torch.tensor(list(map(float, f.readline().split())), dtype=torch.float32)
            for j in range(2):
                for k in range(4):
                    policy[i * 8 + j * 4 + k] = base
                    base = rot_90_cw_policy(base)
                base = horizontal_mirror_image_policy(base)
    return policy

def load_values(filename, n):
    values = torch.empty(n * 8, 1, dtype=torch.float32)
    with open(filename, 'r') as f:
        for i in range(n):
            base = torch.tensor(list(map(float, f.readline().split())), dtype=torch.float32)
            for j in range(8):
                values[i * 8 + j] = base
    return values

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, policy, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.policy = policy
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs_uint8[idx], dtype=torch.uint8).float()
        policy = torch.tensor(self.policy[idx], dtype=torch.float32)
        value = torch.tensor(self.value[idx], dtype=torch.float32)
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, policy, value

inputs = load_128bit_samples("datasets/features.bin")
n = len(inputs) // 8
policies = load_policy("datasets/policies.txt", n)
values = load_values("datasets/values.txt", n)
dataset = Dataset(inputs, policies, values)
training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

BATCH_SIZE = 64
LEARNING_RATE = 0.001
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=16)

from nn import NeuralNetwork, load_model
with open('current_generation.txt', 'r') as f:
    nn_name = f.readline()

print(nn_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name + '.pth')

def loss_fn(prediction, target):
    # could play around with the proportions later
    # policy
    loss = torch.sum(-target[0] * torch.nn.functional.log_softmax(prediction[0], dim=1), dim=1).mean()
    # value
    loss += torch.nn.functional.mse_loss(prediction[1], target[1])
    return loss

optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4, momentum = 0.9)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (input, policy, value) in enumerate(dataloader):
        input, policy, value = input.to(device), policy.to(device), value.to(device)
        prediction = model(input)
        # X, y = model(input)
        # print(X)
        # print("------------------------------------")
        # print(y)
        loss = loss_fn(prediction, (policy, value))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch + 1) % 2000 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for input, policy, value in dataloader:
            input, policy, value = input.to(device), policy.to(device), value.to(device)
            prediction = model(input)
            test_loss += loss_fn(prediction, (policy, value)).item()

    test_loss /= num_batches
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
    print(f"Avg Loss: {test_loss:>8f} \n")
    return test_loss

patience = 5
epochs_without_improvement = 0
best_val_loss = float("inf")
epochs = 100
bestNN = None
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        bestNN = model
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break  # early stopping

torch.save(bestNN.state_dict(), 'models/model_weights_' + str(int(nn_name) + 1) + '.pth')