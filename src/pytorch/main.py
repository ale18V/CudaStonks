import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class StockPredictor(nn.Module):
    def __init__(self, dim_features: int):
        super(StockPredictor, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(dim_features, 128, dtype=torch.float64)
        # Second hidden layer
        self.fc2 = nn.Linear(128, 32, dtype=torch.float64)
        self.fc3 = nn.Linear(32, 1, dtype=torch.float64)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for the output layer
        return x


# Create dummy data for X (input) and Y (output)
DATA_DIR = os.path.join(os.getcwd(), "../../data")
for filename in os.listdir(DATA_DIR):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    Xt = df[["Open", "High", "Low", "Close"]].to_numpy()
    Yt = df["High"].to_numpy()
    s = Xt.shape[1]
    h = 8
    m = len(Xt)
    X = np.zeros(shape=(m-h, h*s))
    for i in range(h):
        X[:, s*i:s*(i+1)] = Xt[i:-h+i]
    Y = Yt[h:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainX, testX = X[:int(m*0.6)], X[int(m*0.6):]
    trainY, testY = Y[:int(m*0.6)], Y[int(m*0.6):]
    trainX = torch.from_numpy(trainX).to(device)
    trainY = torch.from_numpy(trainY).to(device)

    trainX = torch.tensor(trainX, dtype=torch.float64)
    trainY = torch.tensor(trainY, dtype=torch.float64)

    testX = torch.from_numpy(testX).to(device)

    model = StockPredictor(h*s)

    # Move the model to GPU if available
    model.to(device)

    # Loss function (Mean Squared Error)
    criterion = nn.MSELoss(reduction='sum')

    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=1e-5)

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
        predTestY = model(torch.tensor(testX, dtype=torch.float64))
        predTrainY = model(torch.tensor(trainX, dtype=torch.float64))
    predTestY = predTestY.cpu().numpy()
    predTrainY = predTrainY.cpu().numpy()
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
    print(np.mean(np.abs(predTestY - testY)))
