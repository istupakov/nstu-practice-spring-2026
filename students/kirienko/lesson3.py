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


class LinearLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x = None
        self._dw = None
        self._db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self._dw = dy.T @ self._x
        self._db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self._dw is not None and self._db is not None
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        return dy * (self._x > 0)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dy * self._y * (1 - self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self) -> None:
        self._y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        x_shifted = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        self._y = x_shifted - log_sum_exp
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        softmax = np.exp(self._y)
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - softmax * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

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
        return [p for layer in self.layers for p in layer.parameters]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [g for layer in self.layers for g in layer.grad]


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кириенко Илья Владимирович, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> Layer:
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
