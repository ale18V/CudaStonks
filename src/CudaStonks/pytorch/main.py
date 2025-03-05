import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import loaddata
from sklearn.model_selection import train_test_split

from .model import StockPredictor
from .constants import device


for X, Y in loaddata():
    m, n = X.shape

    trainX, testX, trainY, testY = train_test_split(
        np.log(X), np.log(Y), test_size=0.3, shuffle=False)

    trainX = torch.from_numpy(trainX).to(device, dtype=torch.float64)
    trainY = torch.from_numpy(trainY).to(device, dtype=torch.float64)

    testX = torch.from_numpy(testX).to(device, dtype=torch.float64)

    model = StockPredictor(n)

    # Move the model to GPU if available
    model.to(device)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss(reduction='sum')

    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,
                           weight_decay=1e-5)

    def train(model, criterion, optimizer, X, Y, epochs=2000):
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

    model.eval()
    with torch.no_grad():
        predTestY = model(testX).cpu().numpy()
        predTrainY = model(trainX).cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(testY)), testY, 'r', label='Actual')
    axes[1].plot(range(len(trainY)), trainY.cpu(), 'r', label='Actual')
    axes[0].plot(range(len(predTestY)), predTestY, 'b', label='Predicted')
    axes[1].plot(range(len(predTrainY)), predTrainY, 'b', label='Predicted')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs Predicted')
        ax.legend()
    plt.show()
