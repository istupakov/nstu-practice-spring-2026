from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Layer(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def backward(self, dy: np.ndarray) -> np.ndarray: ...

    @property
    def parameters(self) -> Sequence[np.ndarray]: ...

    @property
    def grad(self) -> Sequence[np.ndarray]: ...


class Loss(Protocol):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def backward(self) -> np.ndarray: ...


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.d_weights = (dy.T @ self.x).astype(self.weights.dtype)
        self.d_bias = (dy.sum(axis=0)).astype(self.bias.dtype)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.d_weights, self.d_bias)


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.matrix = x > 0
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.matrix

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * (self.y * (1 - self.y))

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:

        x_max = np.max(x, axis=-1, keepdims=True)

        self.out = (x - x_max) - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
        return self.out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.out) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return tuple(p for layer in self.layers for p in layer.parameters)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return tuple(g for layer in self.layers for g in layer.grad)


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x_back = x - y

        self.size = x.size
        return np.mean(self.x_back**2)

    def backward(self) -> np.ndarray:
        return 2 * self.x_back / self.size


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        self.batch_size = x.shape[0]

        self.x_back = y / x - (1 - y) / (1 - x)
        return -1 * np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self) -> np.ndarray:
        return -1 * (self.x_back / self.batch_size)


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]

        target_log_probs = x[np.arange(self.batch_size), y]

        return np.array(-np.sum(target_log_probs) / self.batch_size)

    def backward(self) -> np.ndarray:

        grad = np.zeros_like(self.x)

        grad[np.arange(self.batch_size), self.y] = -1 / self.batch_size
        return grad


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.y, self.batch_size = y, x.shape[0]

        x_max = np.max(x, axis=-1, keepdims=True)
        log_probs = (x - x_max) - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))

        self.probs = np.exp(log_probs)
        return -np.sum(log_probs[np.arange(self.batch_size), y]) / self.batch_size

    def backward(self) -> np.ndarray:
        grad = self.probs.copy()

        grad[np.arange(self.batch_size), self.y] -= 1
        return grad / self.batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Придатченко Павел Павлович, ПМ-34"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> Layer:
        return LinearLayer(in_features, out_features, rng)

    @staticmethod
    def create_relu_layer() -> Layer:
        return ReLULayer()

    @staticmethod
    def create_sigmoid_layer() -> Layer:
        return SigmoidLayer()

    @staticmethod
    def create_logsoftmax_layer() -> Layer:
        return LogSoftmaxLayer()

    @staticmethod
    def create_model(*layers: Layer) -> Layer:
        return Model(*layers)

    @staticmethod
    def create_mse_loss() -> Loss:
        return MSELoss()

    @staticmethod
    def create_bce_loss() -> Loss:
        return BCELoss()

    @staticmethod
    def create_nll_loss() -> Loss:
        return NLLLoss()

    @staticmethod
    def create_cross_entropy_loss() -> Loss:
        return CrossEntropyLoss()

    @staticmethod
    def train_model(
        model: Layer, loss: Loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int
    ) -> None:

        batch = 0
        exp_num = x.shape[0]
        if batch_size is None:
            batch_size = exp_num

        for _ in range(n_epoch):
            # ind = np.random.permutation(exp_num)

            # x_shuffle = x[ind]
            # y_shuffle = y[ind]

            for batch in range(0, exp_num, batch_size):
                x_batch = x[batch : batch + batch_size]
                y_batch = y[batch : batch + batch_size]

                frwrd = model.forward(x_batch)
                loss.forward(frwrd, y_batch)

                grad = loss.backward()
                model.backward(grad)

                for p, g in zip(model.parameters, model.grad, strict=True):
                    p -= lr * g
