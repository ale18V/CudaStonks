import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neuralnetwork import NeuralNetwork, Optimizer, Adam, ReLU, MSELoss
import numpy as np
import sys
sys.path.append("../")
from dataloader import loaddata  # noqa: E402

for X, Y in loaddata():
    trainXt, testXt, trainYt, testYt = train_test_split(
        X, Y, test_size=0.3, shuffle=False)

    trainX = np.log(trainXt)
    trainY = np.log(trainYt)
    testX = np.log(testXt)
    opt = Adam(eta=0.01)
    model = NeuralNetwork(input_size=trainX.shape[1], hidden_size=40, X=trainX, Y=trainY,
                          activation_fun=ReLU(), loss_fun=MSELoss(), optimizer=opt)

    model.train(lam=1e-3)
    predY = model.predict(testX)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(testYt)), testYt, 'r', label='Actual')
    axes[1].plot(range(len(trainY)), trainYt, 'r', label='Actual')

    axes[0].plot(range(len(predY)), np.exp(predY), 'b', label='Predicted')
    axes[1].plot(range(len(trainY)), np.exp(
        model.predict(trainX)), 'b', label='Predicted')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.setTitle('Actual vs Predicted')
        ax.legend()
    plt.show()
    print(model.W)
    print("Mean error:", np.mean(np.abs(testYt - np.exp(predY))))
