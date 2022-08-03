"""File containing the DQN model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from helpers import device

# Model class
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # Define hidden layers and add Batch Normalization as we're using ReLu
        self.l1 = nn.Linear(in_features=3*3*2, out_features=128)
        #self.bn1 = nn.BatchNorm2d(128)
        self.l2 = nn.Linear(in_features=128, out_features=128)
        #self.bn2 = nn.BatchNorm2d(128)
        self.l3 = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        x = x.view(x.shape[0], 3*3*2)
        x = x.to(device)
        x = F.relu((self.l1(x.float())))
        x = F.relu((self.l2(x)))
        x = self.l3(x)
        return x

    # method to take an action given a state
    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)