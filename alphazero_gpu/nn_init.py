import torch
from torch import nn
import torch.nn.functional as F

# Legacy
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#
#         # Input: 8x8x2 (white pieces, black pieces, valid moves)
#         # Conv layers - maintain 8x8 spatial resolution throughout
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#
#         # Policy head - outputs 64 move probabilities
#         self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)  # 1x1 conv to reduce channels
#         self.policy_fc = nn.Linear(2 * 8 * 8, 64)  # 64 possible moves
#
#         # Value head - outputs single position evaluation
#         self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 conv to reduce channels
#         self.value_fc1 = nn.Linear(1 * 8 * 8, 32)
#         self.value_fc2 = nn.Linear(32, 1)
#
#     def forward(self, x):
#         # x shape: (batch_size, 2, 8, 8)
#
#         # Feature extraction
#         x = F.relu(self.conv1(x))  # (N, 32, 8, 8)
#         x = F.relu(self.conv2(x))  # (N, 32, 8, 8)
#         x = F.relu(self.conv3(x))  # (N, 64, 8, 8)
#         x = F.relu(self.conv4(x))  # (N, 64, 8, 8)
#
#         # Policy head
#         policy = F.relu(self.policy_conv(x))  # (N, 2, 8, 8)
#         policy = torch.flatten(policy, 1)  # (N, 128)
#         policy = self.policy_fc(policy)  # (N, 64)
#         # policy = F.log_softmax(policy, dim=1)
#
#         # Value head
#         value = F.relu(self.value_conv(x))  # (N, 1, 8, 8)
#         value = torch.flatten(value, 1)  # (N, 64)
#         value = F.relu(self.value_fc1(value))  # (N, 32)
#         value = torch.tanh(self.value_fc2(value))  # (N, 1) - bounded [-1, 1]
#
#         return policy, value
    
class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.input_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        self.res_blocks = nn.Sequential(
            *[BottleneckBlock(64) for _ in range(20)]
        )

        self.policy_conv = nn.Conv2d(64, 8, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 8 * 8, 64)

        self.value_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.input_conv(x)))

        x = self.res_blocks(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class BottleneckBlock2(nn.Module):
    def __init__(self, channels):
        super(BottleneckBlock2, self).__init__()
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


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()

        self.input_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        # 16 bottleneck blocks -> 48 conv layers
        self.res_blocks = nn.Sequential(
            *[BottleneckBlock2(64) for _ in range(16)]
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, policy, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.policy = policy
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = self.inputs_uint8[idx].detach().float()
        policy = self.policy[idx].detach()
        value = self.value[idx].detach()
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, policy, value

class NeuralNetworkNNUE(torch.nn.Module):
    # Need to experiment to find a better NN structure
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(128, 256)
        self.layer2 = torch.nn.Linear(256, 32)
        self.layer3 = torch.nn.Linear(32, 32)
        self.value = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        value = F.tanh(self.value(x))
        return value

class DatasetNNUE(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs_uint8[idx], dtype=torch.uint8).float()
        value = torch.tensor(self.value[idx], dtype=torch.float32)
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, value

def load_model(model_class, checkpoint_path, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model