import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, out_channels=32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=2)
        self.fc1   = nn.Linear(64, 256)
        self.fc2   = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x