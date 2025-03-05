from numpy.typing import NDArray
from abc import ABC, abstractmethod


class NNActivation(ABC):
    def __call__(self, Z: NDArray) -> NDArray:
        pass

    def derivative(self, Z: NDArray) -> NDArray:
        pass


class NNLoss(ABC):
    def __call__(self, F: NDArray, Y: NDArray) -> float:
        pass

    def derivative(self, F: NDArray, Y: NDArray) -> NDArray:
        pass


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    def step(self, grad: list[NDArray], weights: list[NDArray]) -> list[NDArray]:
        pass


class NeuralNetworkInterface(ABC):
    @abstractmethod
    def generate_weights(self, num_layers: int, dim_layers: list[int]) -> list[NDArray]:
        """Weights to each hidden unit from the inputs (plus the added offset unit)
        W1 is gonna be shape (n+1, nhid)
        W2 is gonna be shape (nhid+1, nhid/2)
        W3 is gonna be shape (1, nhid/2 + 1)
        """
        raise NotImplementedError()

    @abstractmethod
    def forward_propagate(self, X: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        """
        Returns (pre, act)
        Act[k] is at index k (Activations are 0-based indexed)
        Pre[k] is at index k-1 (Preactivations are 1-based indexed)

        Act[k] is a NDArray of shape (m, nk + 1, 1)
        Pre[k-1] is a NDArray of shape (m, nk,   1)
        Where nk is the number of neurons in the layer k
        """
        raise NotImplementedError()

    @abstractmethod
    def backward_propagate(self, X: NDArray, Y: NDArray) -> list[NDArray]:
        """
        Calculates the gradient of the loss for the i-th row of the dataset
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, X: NDArray, Y: NDArray, lam: float, epoch: int = 1000):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError()

    @abstractmethod
    def eval_loss(self, lam):
        raise NotImplementedError()

    @abstractmethod
    def update_loss(self, f, y):
        raise NotImplementedError()
