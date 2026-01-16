import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actor network
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(self._get_conv_output_size(), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)

        # Initialize last layer weights between -3 and 3
        nn.init.uniform_(self.fc6.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc6.bias, -3e-3, 3e-3)

        self.dropout = nn.Dropout(0.2)

    def _get_conv_output_size(self):
        # Compute output size of conv layers
        x = torch.zeros(1, 1, self.args.state_size, self.args.state_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))

        # Rescale output to match max angle
        return x * math.radians(self.args.max_angle)


# Critic network
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # State pathway
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc_state = nn.Linear(self._get_conv_output_size(), 64)

        # Action pathway
        self.fc_action1 = nn.Linear(args.num_actions, 32)
        self.fc_action2 = nn.Linear(32, 16)

        # Combined pathway
        self.fc1 = nn.Linear(64 + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def _get_conv_output_size(self):
        x = torch.zeros(1, 1, self.args.state_size, self.args.state_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return int(np.prod(x.size()))

    def forward(self, state, action):
        xs = F.relu(self.conv1(state))
        xs = self.bn1(xs)
        xs = F.relu(self.conv2(xs))
        xs = self.bn2(xs)
        xs = self.flatten(xs)
        xs = F.relu(self.fc_state(xs))

        xa = F.relu(self.fc_action1(action))
        xa = F.relu(self.fc_action2(xa))

        x = torch.cat([xs, xa], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# Policy function
def policy(actor_model, state, noise_object, args):
    actor_model.eval()
    with torch.no_grad():
        sampled_actions = actor_model(state).squeeze(1).cpu().numpy()
    noise = noise_object()
    sampled_actions = sampled_actions + noise
    legal_action = np.clip(sampled_actions, math.radians(args.min_angle), math.radians(args.max_angle))
    return [np.squeeze(legal_action)]


# Target update (soft update)
def update_target(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
