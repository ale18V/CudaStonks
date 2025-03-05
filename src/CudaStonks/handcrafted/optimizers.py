import numpy as np
from numpy.typing import NDArray
from typing import Any
from .models import Optimizer


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


class GradDescent(Optimizer):
    def __init__(self, eta: float) -> None:
        super().__init__()
        self.eta = eta

    def step(self, grad: list[NDArray], weights: list[NDArray]) -> list[NDArray]:
        for j in range(len(weights)):
            weights[j] -= self.eta * grad[j]
        return weights