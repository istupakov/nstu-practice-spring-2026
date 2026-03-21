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
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.x is not None, "LinearLayer: incorrect forward and backward order"
        self.grad_weights = dy.T @ self.x
        self.grad_bias = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self.grad_weights is not None and self.grad_bias is not None, (
            "LinearLaye: backward must be called before accessing grad"
        )
        return self.grad_weights, self.grad_bias


class ReLULayer(Layer):
    def __init__(self):
        self.logic_massive: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.logic_massive = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.logic_massive  # np.multiply(dy, self.logic_massive, out=dy)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.y is not None, "SigmoidLayer: incorrect forward and backward order"
        return dy * self.y * (1 - self.y)

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
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        y = (x - x_max) - np.log(sum_exp)
        self.y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.y is not None, "LogSoftmaxLayer: incorrect forward and backward order"
        softmax = np.exp(self.y)
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - softmax * sum_dy

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
        return "Ушатов Сергей Максимович, ПМ-31"

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
