import torch.nn as nn
import torch.nn.functional as F

''' Simplified Network used in Atari DQN Paper'''
class DQNFF(nn.Module):
    def __init__(self):
        super(DQNFF, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.fc2 = nn.Linear(30, 150)
        self.fc3 = nn.Linear(150, 30)
        self.head = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x.view(x.size(0), -1))

''' Cartpole Networks '''
class DQNCNN(nn.Module):

    def __init__(self, num_actions):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels=32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(1,1), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(43776, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels=32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(1,1), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(43776, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))