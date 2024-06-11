from typing import Any
import numpy as np
from handcrafted.neuralnetwork import NeuralNetwork
from numba import cuda, jit
from numpy.typing import NDArray

TILE_SIZE = 8


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
    def weight_mult_kernel(W, A, C):
        # Define an array in the shared memory
        # The size and type of the arrays must be known at compile time
        sW = cuda.shared.array(
            shape=(TILE_SIZE, TILE_SIZE), dtype=np.float64)
        sA = cuda.shared.array(
            shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=np.float64)

        x, y, z = cuda.grid(3)
        tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
        p = A.shape[1]

        if x >= C.shape[0] or y >= C.shape[1] or z >= C.shape[2]:
            # Quit if (x, y) is outside of valid C boundary
            return

        # Each thread computes one element in the result matrix.
        # The dot product is chunked into TPB-long segments.
        tmp = 0.
        for i in range(0, p, TILE_SIZE):
            # Preload data into shared memory
            if tx == 0:
                sW[ty, tz] = W[y + i, z]

            sA[tx, ty, tz] = A[x, y, tz + i]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Computes partial product on the shared memory
            for j in range(TILE_SIZE):
                tmp += sW[j, tz] * sA[tx, ty, j]

            # Wait until all threads finish computing
            cuda.syncthreads()

        C[x, y, z] = tmp

    def weight_mult(self, W: NDArray, B: NDArray) -> NDArray:
        """Performs pairwise matrix multiplication on a list of matrices
        @param: A is size (q, p)
        @param: B is size (m, p, d)
        @return: C is size (m, q, d)
        """
        (m, p, d), q = B.shape, W.shape[0]
        C = np.zeros((m, q, d))

        W_device = cuda.to_device(W)
        B_device = cuda.to_device(B)
        C_device = cuda.to_device(C)
        block_size = (TILE_SIZE, TILE_SIZE, TILE_SIZE)
        grid_size = (int(np.ceil(m/block_size[0])),
                     int(np.ceil(q/block_size[1])),
                     int(np.ceil(d/block_size[2])))
        self.weight_mult_kernel[grid_size, block_size](
            W_device, B_device, C_device)

        return C_device.copy_to_host()

    def forward_propagate(self, X: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        act: list[NDArray] = [None for _ in range(self.num_layers)]
        pre: list[NDArray] = [None for _ in range(self.num_layers)]
        m, n = X.shape
        act[0] = np.insert(X, obj=0, values=1, axis=1).reshape(m, n+1, 1)
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
                               Y=Y.reshape(m, n, 1)))

        for i in reversed(range(1, self.num_layers)):
            delta_a[i] = self.weight_mult(self.W[i].T, delta_z[i])
            grad_w[i] = np.sum(delta_z[i] @ act[i].swapaxes(1, -1), axis=0)  # noqa: Transpose the activations
            delta_z[i-1] = self.activation_function.derivative(
                pre[i-1]) * delta_a[i][:, 1:, :]

        grad_w[0] = np.sum(delta_z[0] @ act[0].swapaxes(1, -1), axis=0)
        return grad_w
