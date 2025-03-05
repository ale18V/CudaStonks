import pandas as pd
import os
import numpy as np
from CudaStonks.constants import DATA_DIR


def get_windowed_array(A: np.ndarray, window_size: int):
    m, s = A.shape
    X = np.zeros(shape=(m - window_size, window_size * s))
    for i in range(window_size):
        X[:, s * i: s * (i + 1)] = A[i: -window_size + i]

    return X


def loaddata():
    for filename in os.listdir(DATA_DIR):
        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        df["Target"] = df["High"].shift(-1)

        data = df[["Open", "High", "Low", "Close", "Target"]].to_numpy()

        Xt = data[:, :-1]
        Yt = data[:, -1:]

        h = 8

        X = get_windowed_array(Xt, h)
        Y = Yt[h - 1: -1]
        yield X, Y


def preprocess(X, Y):
    return np.log(X), np.log(Y)
