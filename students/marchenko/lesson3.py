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

        # инициализация
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)

        # кэш для backward
        self._input: np.ndarray | None = None
        self._grad_weights: np.ndarray | None = None
        self._grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._input is not None
        # Градиент по весам:
        self._grad_weights = dy.T @ self._input

        # Градиент по bias:
        self._grad_bias = np.sum(dy, axis=0)

        # Градиент ко входу:
        dx = dy @ self.weights

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self._grad_bias is not None and self._grad_weights is not None
        return [self._grad_weights, self._grad_bias]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = np.asarray(x)
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # Градиент проходит только где input > 0
        assert self._input is not None
        return dy * (self._input > 0)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._output is not None
        return dy * self._output * (1 - self._output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self._output: np.ndarray | None = None
        self._softmax: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)

        self._softmax = exp_x / (sum_exp)
        self._output = (x - x_max) - np.log(sum_exp)

        return self._output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._softmax is not None
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - sum_dy * self._softmax

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)  # Сохраняем все слои

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
        return "Марченко Вячеслав Иванович, ПМ-33"

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
