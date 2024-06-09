from typing import List
import numpy as np
from numpy.typing import NDArray
from numba import cuda, jit, float64

class Optimizer:
    def step(self, grad: List[NDArray]) -> List[NDArray]:
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, eta: float = 0.001):
        self.eta = eta
        self.m = None
        self.v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def step(self, grad: List[NDArray]) -> List[NDArray]:
        if self.m is None:
            self.m = [np.zeros_like(g) for g in grad]
            self.v = [np.zeros_like(g) for g in grad]
        self.t += 1
        new_grad = []
        for i, g in enumerate(grad):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_grad.append(self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon))
        return new_grad

class ReLU:
    def __call__(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

class MSELoss:
    def __call__(self, pred: NDArray, target: NDArray) -> float:
        return np.mean((pred - target) ** 2)

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, X: NDArray, Y: NDArray, activation_fun: ReLU, loss_fun: MSELoss, optimizer: Optimizer):
        self.X = cuda.to_device(X)
        self.Y = cuda.to_device(Y)
        self.activation_fun = activation_fun
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.W = [cuda.to_device(np.random.randn(hidden_size, input_size)), cuda.to_device(np.random.randn(1, hidden_size))]
        self.loss = 0

    @staticmethod
    @cuda.jit
    def matrix_multiply_kernel(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

    @staticmethod
    @cuda.jit
    def transpose_kernel(A, B):
        i, j = cuda.grid(2)
        if i < A.shape[0] and j < A.shape[1]:
            B[j, i] = A[i, j]

    @staticmethod
    @cuda.jit
    def elementwise_subtract_kernel(A, B, C):
        i = cuda.grid(1)
        if i < A.shape[0]:
            C[i] = A[i] - B[i]

    def forward_propagate(self, X: NDArray) -> List[NDArray]:
        if X.shape[1] != self.W[0].shape[1]:
            raise ValueError(f"Input data shape {X.shape} is not aligned with weight shape {self.W[0].shape}")
        z = np.dot(self.W[0].copy_to_host(), X.T)
        a = self.activation_fun(z)
        z2 = np.dot(self.W[1].copy_to_host(), a)
        a2 = z2  # For regression, output is linear
        return [cuda.to_device(z), cuda.to_device(a), cuda.to_device(z2), cuda.to_device(a2)]

    def backward_propagate(self, X: NDArray, Y: NDArray) -> List[NDArray]:
        z, a, z2, a2 = self.forward_propagate(X)
        delta_z = cuda.device_array(a2.shape, dtype=float64)
        self.elementwise_subtract_kernel[32, 1](a2, Y, delta_z)
        grad_w = [cuda.device_array(w.shape, dtype=float64) for w in self.W]
        # Transpose act[1] for correct dimensions
        act1_T = cuda.device_array((a.shape[1], a.shape[0]), dtype=float64)
        self.transpose_kernel[(32, 32), (32, 32)](a, act1_T)
        self.matrix_multiply_kernel[(32, 32), (32, 32)](delta_z, act1_T, grad_w[1])
        # Transpose act[0] for correct dimensions
        act0_T = cuda.device_array((z.shape[1], z.shape[0]), dtype=float64)
        self.transpose_kernel[(32, 32), (32, 32)](z, act0_T)
        self.matrix_multiply_kernel[(32, 32), (32, 32)](delta_z, act0_T, grad_w[0])
        cuda.synchronize()
        return [g.copy_to_host() for g in grad_w]

    def train(self, lam: float, epoch: int = 1000):
        last_eval, cur_eval = -np.inf, None
        for k in range(epoch):
            gradL = self.backward_propagate(self.X, self.Y)
            for j, w in enumerate(self.W):
                gradL[j] += 2 * lam * w.copy_to_host()
            for j, dw in enumerate(self.optimizer.step(gradL)):
                self.W[j] -= cuda.to_device(dw)
            if k % 10 == 0:
                print(k)
                cur_eval = self.eval_loss(lam)
                improvement = abs(cur_eval - last_eval)
                print("Error:", cur_eval)
                print("Improvement", improvement)
                last_eval = cur_eval
                print()
            self.loss = 0

    def predict(self, X: NDArray) -> NDArray:
        pre, _ = self.forward_propagate(cuda.to_device(X))
        return pre[-1].copy_to_host().flatten()

    def update_loss(self, F: NDArray, Y: NDArray) -> None:
        self.loss += self.loss_fun(F, Y)

    def eval_loss(self, lam: float) -> float:
        pre, act = self.forward_propagate(self.X)
        L = self.loss_fun(pre[-1].copy_to_host(), self.Y.copy_to_host())
        L += sum([(w.copy_to_host() ** 2).sum() for w in self.W]) * lam
        return L
