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

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ll = x
        return self.bias + x @ self.weights.T

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dw = dy.T @ self.ll
        self.db = np.sum(dy, axis=0)
        dx = dy @ self.weights
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.dw, self.db)


class ReLULayer(Layer):
    def __init__(self):
        self.lu: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.lu = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        y = np.maximum(self.lu, 0)
        return dy * np.sign(y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.sgm: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sgm = 1 / (1 + np.exp(-x))
        return self.sgm

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.sgm * (1 - self.sgm)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        self.lsm = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        return self.lsm

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.lsm) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
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
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return tuple(grads)


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return (np.sum((y - x) ** 2)) / y.size

    def backward(self) -> np.ndarray:
        return (2 * (self.x - self.y)) / self.y.size


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.n = x.shape[0]
        return -(np.sum(y * np.log(x) + (1 - y) * np.log(1 - x)) / y.size)

    def backward(self) -> np.ndarray:
        return -((self.y - self.x) / (self.x * (1 - self.x))) / self.n


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.n = x.shape[0]

        if y.ndim == 1:
            return -np.sum(x[np.arange(self.n), y.astype(int)]) / self.n
        else:
            return -np.sum(x * y)

    def backward(self) -> np.ndarray:
        if self.y.ndim == 1:
            grad = np.zeros_like(self.x)

            grad[np.arange(self.n), self.y.astype(int)] = -1 / self.n

            return grad
        else:
            return -self.y / self.n


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.n = x.shape[0]

        c = np.max(x, axis=-1, keepdims=True)
        self.lsm = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        if y.ndim == 1:
            return -np.sum(self.lsm[np.arange(self.n), y.astype(int)]) / self.n
        else:
            return -np.sum(self.lsm * y)

    def backward(self) -> np.ndarray:
        if self.y.ndim == 1:
            grad = np.exp(self.lsm)
            grad[np.arange(self.n), self.y.astype(int)] -= 1

            return grad / self.n
        else:
            grad = np.exp(self.lsm)
            grad -= self.y
            return grad / self.n


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старонедов Владимир Эдуардович, ПМ-33"

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
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for _ in range(n_epoch):
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)

                d_loss = loss.backward()
                model.backward(d_loss)

                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
