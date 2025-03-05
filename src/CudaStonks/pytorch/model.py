import torch
import torch.nn as nn


class StockPredictor(nn.Module):
    def __init__(self, dim_features: int):
        super(StockPredictor, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(dim_features, 64, dtype=torch.float64)
        # Second hidden layer
        self.fc2 = nn.Linear(64, 32, dtype=torch.float64)
        self.fc3 = nn.Linear(32, 1, dtype=torch.float64)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for the output layer
        return x