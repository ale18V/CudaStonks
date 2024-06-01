import numpy as np
from numpy.typing import NDArray
import math


def dtanh(X: NDArray) -> NDArray:
    return 1-np.tanh(X)**2


def sigmoid(z: NDArray | float) -> NDArray | float:
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def relu(X: NDArray | float):
    return np.maximum(0, X)


def drelu(X: NDArray):
    return (X > 0).astype(dtype=X.dtype)


def regression(f, y):
    return (f - y)**2


def dregression(f, y):
    return 2*(f - y)


class NeuralNetwork:
    def __init__(self, num_hidden_layers: int, num_cells: int, X: NDArray, Y: NDArray) -> None:
        """
        @param: X is a numpy array of shape (m, n)
        @param: Y is a numpy array of shape (m, 1)
        """
        self.num_layers: int = num_hidden_layers + 1  
        self.n = X.shape[1]
        self.m = X.shape[0]
        self.k = 1  # Y.shape[1]
        self.W: list[NDArray] = self.generate_weights(
            dim_features=self.n, dim_label=self.k, num_layers=self.num_layers, num_hidden_cells=num_cells)
        self.X: NDArray = X
        self.Y: NDArray = Y
        self.loss = 0


    def generate_weights(self, dim_features: int, dim_label: int, num_layers: int, num_hidden_cells: int) -> list[NDArray]:
        """Weights to each hidden unit from the inputs (plus the added offset unit)
        W1 is gonna be shape (n+1, nhid)
        W2 is gonna be shape (nhid+1, nhid/2)
        W3 is gonna be shape (1, nhid/2 + 1)
        """
        W = [None for i in range(num_layers)]

        rows = num_hidden_cells
        cols = dim_features + 1
        for i in range(num_layers - 1):
            W[i] = (np.random.rand(rows, cols)*2-1)*np.sqrt(6.0/(rows+cols+1))
            cols = rows + 1  # Add the constant 1 term to the activations
            rows = rows // 2

        W[-1] = (np.random.rand(dim_label, cols)*2-1) * \
            np.sqrt(6.0/(num_hidden_cells//2+2))

        return W

    def forward_propagate(self, Xi: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        """
        Returns (pre, act)
        Act[k] is at index k (Activations are 0-based indexed)
        Pre[k] is at index k-1 (Preactivations are 1-based indexed)
        """
        act: list[NDArray] = [None for _ in range(self.num_layers)]
        pre: list[NDArray] = [None for _ in range(self.num_layers)]

        act[0] = self.insertone(Xi).reshape(-1, 1)
        for i in range(self.num_layers - 1):
            pre[i] = self.W[i] @ act[i]  # TODO: Matrix multiply
            act[i+1] = self.insertone(self.activation_function(pre[i]))

        pre[-1] = self.W[-1] @ act[-1]

        return pre, act

    def backward_propagate(self, Xi: NDArray, Yi: float):
        """
        Calculates the gradient of the loss for the i-th row of the dataset 
        """
        pre, act = self.forward_propagate(Xi)
        self.update_loss(pre[-1][0], Yi)

        delta_z: list[NDArray] = [None for _ in range(self.num_layers)]
        delta_a: list[NDArray] = [None for _ in range(self.num_layers)]
        grad_w: list[NDArray] = [None for _ in range(self.num_layers)]

        delta_z[-1] = np.array(self.d_loss_function(f=pre[-1]
                               [0], y=Yi)).reshape(1, 1)
        for i in reversed(range(1, self.num_layers)):
            delta_a[i] = self.W[i].T @ delta_z[i]
            grad_w[i] = delta_z[i] @ act[i].T
            delta_z[i -
                    1] = np.multiply(self.d_activation_function(pre[i-1]), act[i][1:])

        grad_w[0] = delta_z[0] @ act[0].T
        return grad_w

    def train(self, lam, precision):
        Eg2 = 1

        it = 0
        stop = False
        last_eval, cur_eval = -math.inf, None
        while not stop:
            gradL = [np.zeros(self.W[i].shape) for i in range(self.num_layers)]
            for i in range(self.m):
                for j, dw in enumerate(self.backward_propagate(self.X[i], self.Y[i])):
                    gradL[j] += dw

            sumofgrad2 = np.sum([np.sum(g**2) for g in gradL])
            Eg2, eta = self.update_eta(Eg2, sumofgrad2)

            for j in range(self.num_layers):
                self.W[j] -= eta*gradL[j]

            it += 1
            if it % 10 == 0:
                print(it)
                cur_eval = self.eval_loss(lam)
                improvement = abs(cur_eval - last_eval)
                print("Error:", cur_eval)
                print("Improvement", improvement)

                stop = cur_eval < last_eval and improvement <= precision
                last_eval = cur_eval
            self.loss = 0

    def predict(self, X: NDArray) -> NDArray:
        Y = np.zeros(len(X), dtype=X.dtype)
        for i in range(len(X)):
            pre, act = self.forward_propagate(self.get_features(X, i))
            Y[i] = pre[-1][0]
        return Y

    def eval_loss(self, lam):
        return self.loss + lam*np.sum([np.sum(w**2) for w in self.W])

    def update_loss(self, f, y):
        self.loss += self.loss_function(f, y)

    def loss_function(self, f: float, y: float):
        return regression(f, y)
        # return -np.log(sigmoid(y*f))

    def d_loss_function(self, f: float, y: float):
        return dregression(f, y)
        # return -sigmoid(-y*f)*y

    def activation_function(self, pre: NDArray):
        return relu(pre)
        # return np.tanh(pre)

    def d_activation_function(self, X: NDArray):
        return drelu(X)
        #return 1-np.tanh(X)**2

    def insertone(self, X: NDArray):
        """Adds a '1' at the start of a 1-d array"""
        return np.insert(X, 0, 1, axis=0)

    def update_eta(self, Eg2, sumofgrad2):
        """When you need "eta" (the step size), do this after you have already
        calculated the gradient (but before you take a step):
        [where "sumofgrad2" is the sum of the squares of each element of the gradient
        that is, you square *all* of the gradient values and then add them all together]
        [recall that this is the gradient of the full loss function that includes every
        example *and* the regularizer term]"""
        Eg2 = 0.9*Eg2 + 0.1*sumofgrad2
        eta = 0.01/(np.sqrt((1e-10+Eg2)))
        return Eg2, eta
