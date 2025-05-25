import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models

def load_136bit_samples(filename):
    with open(filename, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    # Unpack all bits
    print(raw[:100])
    print(len(raw))
    all_bits = np.unpackbits(raw)

    # Each sample = 136 bits (129 inputs + 5 unused + 2 label)
    print(all_bits.size)
    num_samples = all_bits.size // 136
    all_bits = all_bits[:num_samples * 136]

    data = all_bits.reshape((num_samples, 136))

    inputs = data[:, :129]

    labels = data[:, 134:]
    labels = np.packbits(labels, axis = -1)
    return inputs, labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, labels_uint8, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.labels_uint8 = labels_uint8
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        X = torch.tensor(self.inputs_uint8[idx], dtype=torch.uint8)
        y = torch.tensor(self.labels_uint8[idx], dtype=torch.uint8)
        X = X.float()
        y = y.float() / 128

        if self.transform:
            X = self.transform(X)

        return X, y

nnue_name = input()
inputs, labels = load_136bit_samples("datasets/data_" + nnue_name + ".bin")
# experimental data
# inputs, labels = load_136bit_samples('data2.bin')

dataset = Dataset(inputs, labels)
training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)

learning_rate = 0.3
batch_size = 64

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch % 2000 == 0:
      loss, current = loss.item(), batch * batch_size + len(X)
      print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")
    #   print(pred)
    #   print(loss)
    #   print(X)
    #   print(y)

def test_loop(dataloader, model, loss_fn):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
    #   print(type(pred))
    #   print(type(y))
    #   print(pred)
    #   print(y)
      correct += (abs(pred - y) < 0.5).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
#   print(f"Avg Loss: {test_loss:>8f} \n")
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
  return test_loss

patience = 2
epochs_without_improvement = 0
best_val_loss = float("inf")
epochs = 5
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

print("Done!")

print(model)
state_dict = model.state_dict()

params = {k: v.cpu().numpy() for k, v in state_dict.items()}
print(type(params['linear_relu_stack.0.weight'][0]))
print(len(params['linear_relu_stack.0.weight'])) # rows
print(len(params['linear_relu_stack.0.weight'][0])) # columns

open('weights.txt', 'w').close()
open('biases.txt', 'w').close()
for x in range(3):
    y = 2 * x
    weights = params['linear_relu_stack.' + str(y) + '.weight']
    weights = weights.flatten()
    file1 = open("weights.txt", "a+")
    file1.write(" ".join(str(x) for x in weights.tolist()))
    file1.write('\n')
    file1.close()

    bias = params['linear_relu_stack.' + str(y) + '.bias']
    file2 = open("biases.txt", "a+")
    file2.write(" ".join(str(x) for x in bias.tolist()))
    file2.write('\n')
    file2.close()

torch.save(model.state_dict(), 'model_weights.pth')