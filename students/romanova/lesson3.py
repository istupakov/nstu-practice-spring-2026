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


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.W = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.b = rng.uniform(-k, k, out_features).astype(np.float32)

        self.W_grad: np.ndarray
        self.b_grad: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.W_grad = dy.T @ self.x
        self.b_grad = np.sum(dy, axis=0)
        return dy @ self.W

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.W, self.b)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.W_grad, self.b_grad)


class ReLULayer(Layer):
    def __init__(self):
        self.input_data: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_data = x
        return np.where(x > 0, x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        mask = (self.input_data > 0).astype(np.float32)
        return dy * mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.activation: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation = 1.0 / (1.0 + np.exp(-x))
        return self.activation

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.activation * (1 - self.activation)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.log_probs: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_centered = x - np.max(x, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x_centered), axis=-1, keepdims=True))
        self.log_probs = x_centered - log_sum_exp
        return self.log_probs

    def backward(self, dy: np.ndarray) -> np.ndarray:
        probs = np.exp(self.log_probs)
        return dy - probs * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
        self.network = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x
        for layer in self.network:
            current = layer.forward(current)
        return current

    def backward(self, dy: np.ndarray) -> np.ndarray:
        grad = dy
        for layer in reversed(self.network):
            grad = layer.backward(grad)
        return grad

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        result = []
        for layer in self.network:
            result.extend(layer.parameters)
        return tuple(result)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        result = []
        for layer in self.network:
            result.extend(layer.grad)
        return tuple(result)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Романова Валерия Сергеевна, ПМ-34"

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
