import numpy as np
from numpy.typing import NDArray
from .models import NNActivation


class TanH(NNActivation):
    def __call__(self, Z: NDArray) -> NDArray:
        return np.tanh(Z)

    def derivative(self, Z: NDArray) -> NDArray:
        return 1 - np.tanh(Z) ** 2


class ReLU(NNActivation):
    def __call__(self, Z: NDArray) -> NDArray:
        return np.maximum(0, Z)

    def derivative(self, Z: NDArray) -> NDArray:
        return (Z > 0).astype(dtype=Z.dtype)
