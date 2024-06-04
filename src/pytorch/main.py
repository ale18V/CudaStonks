import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import loaddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class StockPredictor(nn.Module):
    def __init__(self, dim_features: int):
        super(StockPredictor, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(dim_features, 40, dtype=torch.float64)
        # Second hidden layer
        self.fc2 = nn.Linear(40, 20, dtype=torch.float64)
        self.fc3 = nn.Linear(20, 1, dtype=torch.float64)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for the output layer
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for X, Y in loaddata():
    m, n = X.shape

    trainXt, testXt, trainYt, testYt = train_test_split(
        X, Y, test_size=0.3, shuffle=False)

    trainX = torch.from_numpy(np.log(trainXt)).to(device, dtype=torch.float64)
    trainY = torch.from_numpy(np.log(trainYt)).to(device, dtype=torch.float64)

    testX = torch.from_numpy(np.log(testXt)).to(device, dtype=torch.float64)

    model = StockPredictor(n)

    # Move the model to GPU if available
    model.to(device)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss(reduction='sum')

    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           weight_decay=1e-4)

    def train(model, criterion, optimizer, X, Y, epochs=1000):
        model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    # Train the model
    train(model, criterion, optimizer, trainX, trainY, epochs=2000)
    model.eval()
    with torch.no_grad():
        predTestY = model(testX).cpu().numpy()
        predTrainY = model(trainX).cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(testYt)), testYt, 'r', label='Actual')
    axes[1].plot(range(len(trainY)), trainYt, 'r', label='Actual')
    axes[0].plot(range(len(predTestY)), np.exp(
        predTestY), 'b', label='Predicted')
    axes[1].plot(range(len(predTrainY)), np.exp(
        predTrainY), 'b', label='Predicted')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs Predicted')
        ax.legend()
    plt.show()
