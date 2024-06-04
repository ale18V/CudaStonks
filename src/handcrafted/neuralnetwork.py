import numpy as np
from numpy.typing import NDArray
import math


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

    def step(self, grad: list[NDArray]) -> list[NDArray]:
        return []


class NeuralNetwork:
    def __init__(self, num_hidden_layers: int, num_cells: int, X: NDArray, Y: NDArray, optimizer: Optimizer,
                 activation_fun: NNActivation, loss_fun: NNLoss) -> None:
        """
        @param: X is a numpy array of shape (m, n)
        @param: Y is a numpy array of shape (m, 1)
        """
        self.num_layers: int = num_hidden_layers + 1
        self.m, self.n = X.shape
        self.k = 1  # Y.shape[1]
        self.W: list[NDArray] = self.generate_weights(
            dim_features=self.n, dim_label=self.k, num_layers=self.num_layers, num_hidden_cells=num_cells)
        self.X: NDArray = X
        self.Y: NDArray = Y
        self.loss = 0
        self.loss_function = loss_fun
        self.activation_function = activation_fun
        self.optimizer = optimizer

    def generate_weights(self, dim_features: int, dim_label: int, num_layers: int, num_hidden_cells: int) -> list[NDArray]:
        """Weights to each hidden unit from the inputs (plus the added offset unit)
        W1 is gonna be shape (n+1, nhid)
        W2 is gonna be shape (nhid+1, nhid/2)
        W3 is gonna be shape (1, nhid/2 + 1)
        """
        W = [None for i in range(num_layers)]

        rows = num_hidden_cells
        cols = dim_features + 1

        def kaiming_init(p, d):
            """p is the size of the previous layer
            d is the size of the current layer
            """
            return np.random.normal(size=p*d).reshape(p, d)*math.sqrt(2.0/p)

        for i in range(num_layers - 1):
            W[i] = kaiming_init(rows, cols)
            cols = rows + 1  # Add the constant 1 term to the activations
            rows = rows // 2

        W[-1] = kaiming_init(dim_label, cols)
        return W

    def forward_propagate(self, X: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        """
        Returns (pre, act)
        Act[k] is at index k (Activations are 0-based indexed)
        Pre[k] is at index k-1 (Preactivations are 1-based indexed)

        Act[k] is a NDArray of shape (m, nk + 1, 1)
        Pre[k-1] is a NDArray of shape (m, nk,   1)
        Where nk is the number of neurons in the layer k
        """
        act: list[NDArray] = [None for _ in range(self.num_layers)]
        pre: list[NDArray] = [None for _ in range(self.num_layers)]

        m, n = X.shape
        act[0] = np.insert(X, obj=0, values=1, axis=1).reshape(m, n+1, 1)
        for i in range(self.num_layers - 1):
            pre[i] = self.W[i] @ act[i]  # TODO: Matrix multiply
            act[i+1] = np.insert(self.activation_function(pre[i]),
                                 obj=0, values=1, axis=1)

        pre[-1] = self.W[-1] @ act[-1]

        return pre, act

    def backward_propagate(self, X: NDArray, Y: NDArray) -> list[NDArray]:
        """
        Calculates the gradient of the loss for the i-th row of the dataset
        """
        pre, act = self.forward_propagate(X)
        m, n = Y.shape

        self.update_loss(pre[-1], Y)

        delta_z: list[NDArray] = [None for _ in range(self.num_layers)]
        delta_a: list[NDArray] = [None for _ in range(self.num_layers)]
        grad_w: list[NDArray] = [None for _ in range(self.num_layers)]
        delta_z[-1] = np.array(self.loss_function.derivative(F=pre[-1],
                               Y=Y.reshape(m, n, 1)))

        for i in reversed(range(1, self.num_layers)):
            delta_a[i] = self.W[i].T @ delta_z[i]
            grad_w[i] = np.sum(delta_z[i] @ act[i].swapaxes(1, -1), axis=0)  # noqa: Transpose the activations
            delta_z[i -
                    1] = np.multiply(self.activation_function.derivative(pre[i-1]), act[i][:, 1:])

        grad_w[0] = np.sum(delta_z[0] @ act[0].swapaxes(1, -1), axis=0)
        return grad_w

    def train(self, lam: float, epoch: int = 1000):
        last_eval, cur_eval = -math.inf, None
        for k in range(epoch):
            gradL = self.backward_propagate(self.X, self.Y)
            for j, w in enumerate(self.W):
                gradL[j] += 2*lam*w

            for j, dw in enumerate(self.optimizer.step(gradL)):
                self.W[j] -= dw

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
        pre, act = self.forward_propagate(X)
        return pre[-1].squeeze()

    def eval_loss(self, lam):
        return self.loss + lam*np.sum([np.sum(w**2) for w in self.W])

    def update_loss(self, f, y):
        self.loss += self.loss_function(f, y)
