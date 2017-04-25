import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, out_channels=32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(3,3), stride=1)
        self.fc1   = nn.Linear(64*7*7, 512)
        self.fc2   = nn.Linear(512, 64)
        self.fc3   = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        print "HERE"
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def __init__(self):
    #     super(DQN, self).__init__()
    #     self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    #     self.bn2 = nn.BatchNorm2d(32)
    #     self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    #     self.bn3 = nn.BatchNorm2d(32)
    #     self.head = nn.Linear(448, 2)

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     return self.head(x.view(x.size(0), -1))