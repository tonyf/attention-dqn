import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, out_channels=32, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(1,1), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(576, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))