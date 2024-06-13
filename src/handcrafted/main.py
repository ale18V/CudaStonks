import matplotlib.pyplot as plt
from typing import Any
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from neuralnetwork import NNActivation, NNLoss, NeuralNetwork, Optimizer
import numpy as np
import sys
sys.path.append("../")
from dataloader import loaddata  # noqa: E402
from params import learning_rate, regularization_strength, epochs, layers_size  # noqa: E402


class Adagrad(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self.Eg2 = 1

    def step(self, grad: list[np.ndarray[Any, np.dtype]], weights: list[NDArray]) -> list[np.ndarray[Any, np.dtype]]:
        sumofgrad2 = np.sum([np.sum(g**2) for g in grad])
        self.update_eta(sumofgrad2)
        for j in range(len(weights)):
            weights[j] -= self.eta * grad[j]
        return weights

    def update_eta(self, sumofgrad2):
        """When you need "eta" (the step size), do this after you have already
        calculated the gradient (but before you take a step):
        [where "sumofgrad2" is the sum of the squares of each element of the gradient
        that is, you square *all* of the gradient values and then add them all together]
        [recall that this is the gradient of the full loss function that includes every
        example *and* the regularizer term]"""
        self.Eg2 = 0.9*self.Eg2 + 0.1*sumofgrad2
        self.eta = 0.01/(np.sqrt((1e-10+self.Eg2)))


class Adam(Optimizer):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__()
        self.m: list[np.ndarray[Any, np.dtype]] = None
        self.v: list[np.ndarray[Any, np.dtype]] = None
        self.beta1 = np.float64(beta1)
        self.beta2 = np.float64(beta2)
        self.eta = np.float64(eta)
        self.epsilon = np.float64(epsilon)
        self.t = 1

    def step(self, grad: list[np.ndarray[Any, np.dtype]], weights: list[NDArray]) -> list[np.ndarray[Any, np.dtype]]:
        if not self.m or not self.v:
            self.m = [np.zeros(shape=g.shape, dtype=g.dtype) for g in grad]
            self.v = [np.zeros(shape=g.shape, dtype=g.dtype) for g in grad]

        m_unbiased = [None for _ in range(len(grad))]
        v_unbiased = [None for _ in range(len(grad))]
        for j, g in enumerate(grad):
            self.m[j] = self.beta1*self.m[j] + (1-self.beta1)*g
            self.v[j] = self.beta2*self.v[j] + (1-self.beta2)*g**2
            m_unbiased[j] = self.m[j]/(1-self.beta1**self.t)
            v_unbiased[j] = self.v[j]/(1-self.beta2**self.t)
            weights[j] -= self.eta * (m_unbiased[j] /
                                      (np.sqrt(v_unbiased[j]) + self.epsilon))
        self.t += 1
        return weights


class ReLU(NNActivation):
    def __call__(self, Z: NDArray) -> NDArray:
        return np.maximum(0, Z)

    def derivative(self, Z: NDArray) -> NDArray:
        return (Z > 0).astype(dtype=Z.dtype)


class TanH(NNActivation):
    def __call__(self, Z: NDArray) -> NDArray:
        return np.tanh(Z)

    def derivative(self, Z: NDArray) -> NDArray:
        return 1 - np.tanh(Z)**2


class MSELoss(NNLoss):
    def __call__(self, F: NDArray, Y: NDArray) -> float:
        return ((F - Y)**2).sum()

    def derivative(self, F: NDArray, Y: NDArray) -> NDArray:
        return 2*(F - Y)


class LogisticLoss(NNLoss):
    def __call__(self, F: NDArray, Y: NDArray) -> float:
        return -np.log(self.__sigmoid(Y*F)).sum()

    def derivative(self, F: NDArray, Y: NDArray) -> NDArray:
        return -self.__sigmoid(-Y*F)*Y

    def __sigmoid(self, z: NDArray | float):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))


class GradDescent(Optimizer):
    def __init__(self, eta: float) -> None:
        super().__init__()
        self.eta = eta

    def step(self, grad: list[NDArray], weights: list[NDArray]) -> list[NDArray]:
        for j in range(len(weights)):
            weights[j] -= self.eta * grad[j]
        return weights


for X, Y in loaddata():
    trainX, testX, trainY, testY = train_test_split(
        np.log(X), np.log(Y), test_size=0.3, shuffle=False)

    m, n = X.shape

    opt = Adam(eta=learning_rate)
    model = NeuralNetwork(dim_features=n, dim_label=1, dim_hidden_layers=layers_size,
                          activation_fun=ReLU(),
                          loss_fun=MSELoss(), optimizer=opt)
    
    import time
    print("Training model")
    start = time.time()
    model.train(trainX, trainY, lam=regularization_strength, epoch=epochs)
    end = time.time()
    print("Time elapsed:", end - start)
    print()
    predTestY = model.predict(testX)
    predY = model.predict(trainX)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(testY)), testY, c='r', label='Actual')
    axes[1].plot(range(len(trainY)), trainY, c='r', label='Actual')

    axes[0].scatter(range(len(predTestY)), predTestY, c='b', label='Predicted')
    axes[1].scatter(range(len(predY)), predY, c='b', label='Predicted')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs Predicted')
        ax.legend()
    plt.show()

    fig.savefig('actual_vs_predicted.png')
