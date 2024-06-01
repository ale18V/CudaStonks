from neuralnetwork import NeuralNetwork
import pandas as pd
import os
import numpy as np

HIDDEN_LAYERS = 3
HIDDEN_CELLS = 32
DATA_DIR = os.path.join(os.getcwd(), "../data")


def get_windowed_array(A: np.ndarray, window_size: int):
    m, s = A.shape
    X = np.zeros(shape=(m-window_size, window_size*s))
    for i in range(window_size):
        X[:, s*i:s*(i+1)] = Xt[i:-window_size+i]

    return X

for filename in os.listdir(DATA_DIR):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    Xt = df[["Open", "High", "Low", "Close"]].to_numpy()
    Yt = df["High"].to_numpy()

    h = 8
    m, s = Xt.shape

    X = get_windowed_array(Xt, h)
    Y = Yt[h:]
    trainX, testX = X[int(m*0.6):], X[:int(m*0.6)]
    trainY, testY = Y[int(m*0.6):], Y[:int(m*0.6)]

    model = NeuralNetwork(2, 16, trainX, trainY)

    model.train(1e-5, 1e-5)
    predY = model.predict(testX)
    print("Mean error:", np.mean(np.abs(testY - predY)))
