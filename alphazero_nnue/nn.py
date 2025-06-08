import torch
from torch import nn
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    # Need to experiment to find a better NN structure
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(128, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.policy = nn.Linear(128, 64)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        policy = F.softmax(self.policy(x), dim=0)
        value = F.tanh(self.value(x))
        return policy, value

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
# model = NeuralNetwork().to(device)

# state_dict = model.state_dict()
# torch.save(model.state_dict(), './models/model_weights_1.pth')

def load_model(model_class, checkpoint_path, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

