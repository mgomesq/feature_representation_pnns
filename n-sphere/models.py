import torch
from torch import nn

class MyNetwork(nn.Module):
    def __init__(self, combined=False):
        # call constructor from superclass
        super().__init__()
        
        # define network layers
        if combined:
            self.fc1 = nn.Linear(2, 6, dtype=torch.float)
        else:
            self.fc1 = nn.Linear(4, 6, dtype=torch.float)
            
        self.fc2 = nn.Linear(6, 6, dtype=torch.float)
        self.fc2 = nn.Linear(6, 1, dtype=torch.float)
        
    def forward(self, x):
        # define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
