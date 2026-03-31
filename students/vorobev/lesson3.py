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
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self.x = None
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dw = dy.T @ self.x
        self.db = dy.sum(axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self.dw, self.db]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self.mask: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return tuple()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return tuple()


class SigmoidLayer(Layer):
    def __init__(self):
        self.out: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.out * (1 - self.out)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return tuple()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return tuple()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self.sm = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        self.sm = exp_x / sum_exp
        return x - x_max - np.log(sum_exp)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - self.sm * dy.sum(axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return tuple()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return tuple()


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
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Воробьев Никита Александрович, ПМ-31"

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
