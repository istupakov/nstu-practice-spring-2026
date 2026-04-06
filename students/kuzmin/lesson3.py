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
        self.input = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.grad_w = dy.T @ self.input
        self.grad_b = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self.grad_w, self.grad_b]


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * (self.input > 0)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sig = 1 / (1 + np.exp(-x))
        return self.sig

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.sig * (1 - self.sig)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        self.y = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.y) * np.sum(dy, axis=-1, keepdims=True)

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
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.mean(np.square(y - x))

    def backward(self) -> np.ndarray:
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))

    def backward(self) -> np.ndarray:
        batch_size = self.x.shape[0]
        return -(self.y - self.x) / (self.x * (1 - self.x)) / batch_size


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return -np.mean(x[np.arange(x.shape[0]), y])

    def backward(self) -> np.ndarray:
        batch_size = self.x.shape[0]
        dx = np.zeros_like(self.x)
        dx[np.arange(self.x.shape[0]), self.y] = -1 / batch_size
        return dx


class CrossEntropyLoss(Loss):
    def log_softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.batch_size = x.shape[0]
        self.hot_y = np.zeros_like(x)
        self.hot_y[np.arange(x.shape[0]), y] = 1
        self.logprobs = self.log_softmax(x)
        return -np.sum(self.logprobs * self.hot_y) / x.shape[0]

    def backward(self) -> np.ndarray:
        return (np.exp(self.logprobs) - self.hot_y) / self.batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузьмин Александр Андреевич, ПМ-35"

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
        for _ in range(n_epoch):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())
                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g
