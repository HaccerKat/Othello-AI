import torch
import torch.nn.functional as F
class NeuralNetwork(torch.nn.Module):
    # Need to experiment to find a better NN structure
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(128, 256)
        self.policy_hidden1 = torch.nn.Linear(256, 256)
        self.policy_hidden2 = torch.nn.Linear(256, 128)
        self.policy = torch.nn.Linear(128, 64)

        self.value_hidden1 = torch.nn.Linear(256, 128)
        self.value = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared(x))
        y = x
        x = F.relu(self.policy_hidden1(x))
        x = F.relu(self.policy_hidden2(x))
        policy = self.policy(x)

        y = F.relu(self.value_hidden1(y))
        value = F.tanh(self.value(y))
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

