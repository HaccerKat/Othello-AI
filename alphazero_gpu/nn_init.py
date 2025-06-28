import torch
from torch import nn
import torch.nn.functional as F

# class NeuralNetwork(torch.nn.Module):
#     # Need to experiment to find a better NN structure
#     def __init__(self):
#         super().__init__()
#         self.shared = torch.nn.Linear(128, 256)
#         self.policy_hidden1 = torch.nn.Linear(256, 256)
#         self.policy_hidden2 = torch.nn.Linear(256, 128)
#         self.policy = torch.nn.Linear(128, 64)

#         self.value_hidden1 = torch.nn.Linear(256, 128)
#         self.value = torch.nn.Linear(128, 1)

#     def forward(self, x):
#         x = F.relu(self.shared(x))
#         y = x
#         x = F.relu(self.policy_hidden1(x))
#         x = F.relu(self.policy_hidden2(x))
#         policy = self.policy(x)

#         y = F.relu(self.value_hidden1(y))
#         value = F.tanh(self.value(y))
#         return policy, value

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Input: 8x8x2 (white pieces, black pieces, valid moves)
        # Conv layers - maintain 8x8 spatial resolution throughout
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head - outputs 64 move probabilities
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)  # 1x1 conv to reduce channels
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)  # 64 possible moves

        # Value head - outputs single position evaluation
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 conv to reduce channels
        self.value_fc1 = nn.Linear(1 * 8 * 8, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, 2, 8, 8)

        # Feature extraction
        x = F.relu(self.conv1(x))  # (N, 32, 8, 8)
        x = F.relu(self.conv2(x))  # (N, 32, 8, 8)
        x = F.relu(self.conv3(x))  # (N, 64, 8, 8)
        x = F.relu(self.conv4(x))  # (N, 64, 8, 8)

        # Policy head
        policy = F.relu(self.policy_conv(x))  # (N, 2, 8, 8)
        policy = torch.flatten(policy, 1)  # (N, 128)
        policy = self.policy_fc(policy)  # (N, 64)
        # policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_conv(x))  # (N, 1, 8, 8)
        value = torch.flatten(value, 1)  # (N, 64)
        value = F.relu(self.value_fc1(value))  # (N, 32)
        value = torch.tanh(self.value_fc2(value))  # (N, 1) - bounded [-1, 1]

        return policy, value
    
class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity  # Skip connection
        return F.relu(out)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.input_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        # 16 bottleneck blocks → 48 conv layers
        self.res_blocks = nn.Sequential(
            *[BottleneckBlock(64) for _ in range(16)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8 * 1, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.input_conv(x)))

        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
#
class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, policy, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.policy = policy
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = self.inputs_uint8[idx].clone().detach().float()
        policy = self.policy[idx].clone().detach()
        value = self.value[idx].clone().detach()
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, policy, value

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