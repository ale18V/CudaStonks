import numpy as np
from numpy.typing import NDArray
from .models import NNLoss


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