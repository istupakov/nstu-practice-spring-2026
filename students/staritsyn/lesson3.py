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
        self.x: np.ndarray | None = None
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_bias: np.ndarray = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.grad_weights = dy.T @ self.x
        self.grad_bias = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return self.grad_weights, self.grad_bias


class ReLULayer(Layer):
    def __init__(self):
        self.lm: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.lm = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.lm

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.sgm: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.sgm = 1 / (1 + np.exp(-x))
        return self.sgm

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.sgm is None:
            raise ValueError("forward must be called before backward")
        return dy * self.sgm * (1 - self.sgm)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        self.y = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.y is None:
            raise ValueError("forward must be called before backward")
        return dy - (np.exp(self.y) * np.sum(dy, axis=-1, keepdims=True))

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
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [p for layer in self.layers for p in layer.parameters]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [g for layer in self.layers for g in layer.grad]


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старицын Марк Вадимович, ПМ-35"

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
