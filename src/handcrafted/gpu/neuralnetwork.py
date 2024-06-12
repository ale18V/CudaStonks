from typing import Any
import numpy as np
from handcrafted.neuralnetwork import NeuralNetwork
from numba import cuda, jit
from numpy.typing import NDArray

BLOCK_SIZE = 16


class NNActivation:
    def __call__(self, Z: NDArray) -> NDArray:
        pass

    def derivative(self, Z: NDArray) -> NDArray:
        pass


class NNLoss:
    def __call__(self, F: NDArray, Y: NDArray) -> float:
        pass

    def derivative(self, F: NDArray, Y: NDArray) -> NDArray:
        pass


class Optimizer:
    def __init__(self) -> None:
        pass

    def step(self, grad: list[NDArray], weights: list[NDArray]) -> list[NDArray]:
        return []


class NeuralNetworkGPU(NeuralNetwork):
    @staticmethod
    @cuda.jit
    def matmul_kernel(A: NDArray, B: NDArray, C: NDArray):
        """Performs matrix multiplication
        @param: A a matrix of size (m, k)
        @param: B a matrix of size (k, n)
        @return: C a list of the matrix products of size (m, n)
        """
        # Define shared memory arrays
        cacheA = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
        cacheB = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)

        # Calculate thread indices
        col, row = cuda.grid(2)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # Initialize the accumulation variable
        tmp = 0.0
        (m, k), n = A.shape, B.shape[1]

        # Loop over the tiles
        for p in range(0, k, BLOCK_SIZE):
            # Load elements into shared memory
            if row < m and (p + tx) < k:
                cacheA[ty, tx] = A[row, p + tx]
            else:
                cacheA[ty, tx] = 0.0

            if col < n and (p + ty) < k:
                cacheB[ty, tx] = B[p + ty, col]
            else:
                cacheB[ty, tx] = 0.0

            # Synchronize threads to ensure all data is loaded
            cuda.syncthreads()

            # Compute partial product
            for i in range(BLOCK_SIZE):
                tmp += cacheA[ty, i] * cacheB[i, tx]

            # Synchronize threads to ensure computation is complete
            cuda.syncthreads()

        # Write the result to the output matrix
        if row < m and col < n:
            C[row, col] = tmp

    def weight_mult(self, W: NDArray, M: NDArray) -> NDArray:
        """
        @param: W is the weight matrix of (q, p)
        @param: M is an array of m vectors to be multiplied to W
        @return: C is size (m, q) 
        and contains the list of matrix products between W and every vector in M.
        """
        m, q = M.shape[0], W.shape[0]
        C = np.zeros((m, q))

        W_device = cuda.to_device(W.T)
        M_Device = cuda.to_device(M)
        C_device = cuda.to_device(C)

        block_size = (BLOCK_SIZE, BLOCK_SIZE)

        grid_size = (int(np.ceil(m/block_size[0])),
                     int(np.ceil(q/block_size[1])))

        self.matmul_kernel[grid_size, block_size](
            M_Device, W_device, C_device)

        return C_device.copy_to_host()

    def forward_propagate(self, X: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        act: list[NDArray] = [None for _ in range(self.num_layers)]
        pre: list[NDArray] = [None for _ in range(self.num_layers)]
        m, n = X.shape
        act[0] = np.insert(X, obj=0, values=1, axis=1).reshape(m, n+1)
        for i in range(self.num_layers - 1):
            # TODO: Matrix multiply
            pre[i] = self.weight_mult(self.W[i], act[i])
            act[i+1] = np.insert(self.activation_function(pre[i]),
                                 obj=0, values=1, axis=1)

        pre[-1] = self.weight_mult(self.W[-1], act[-1])

        return pre, act

    def backward_propagate(self, X: NDArray, Y: NDArray) -> list[NDArray]:
        """
        Calculates the gradient of the loss for the i-th row of the dataset
        """
        pre, act = self.forward_propagate(X)
        m, n = Y.shape

        self.update_loss(pre[-1].reshape(m, -1), Y)

        delta_z: list[NDArray] = [None for _ in range(self.num_layers)]
        delta_a: list[NDArray] = [None for _ in range(self.num_layers)]
        grad_w: list[NDArray] = [None for _ in range(self.num_layers)]
        delta_z[-1] = np.array(self.loss_function.derivative(F=pre[-1],
                               Y=Y.reshape(m, n)))

        for i in reversed(range(1, self.num_layers)):
            delta_a[i] = self.weight_mult(self.W[i].T, delta_z[i])
            grad_w[i] = np.sum(delta_z[i] @ act[i].swapaxes(1, -1), axis=0)  # noqa: Transpose the activations
            delta_z[i-1] = self.activation_function.derivative(
                pre[i-1]) * delta_a[i][:, 1:, :]

        grad_w[0] = np.sum(delta_z[0] @ act[0].swapaxes(1, -1), axis=0)
        return grad_w
