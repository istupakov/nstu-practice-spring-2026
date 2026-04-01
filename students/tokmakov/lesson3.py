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

        limit = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-limit, limit, out_features).astype(np.float32)

        self._x: np.ndarray = np.empty(0, dtype=np.float32)
        self._weights_grad: np.ndarray = np.zeros_like(self.weights)
        self._bias_grad: np.ndarray = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx = dy @ self.weights
        self._weights_grad = dy.T @ self._x
        self._bias_grad = np.sum(dy, axis=0)

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self._weights_grad, self._bias_grad]


class ReLULayer:
    def __init__(self) -> None:
        self._mask: np.ndarray = np.empty(0, dtype=bool)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = 1 / (1 + np.exp(-x))
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._y * (1 - self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis
        self._softmax: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self.axis, keepdims=True)
        shifted = x - x_max
        exp_shifted = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_shifted, axis=self.axis, keepdims=True))

        out = shifted - log_sum_exp
        self._softmax = np.exp(out)
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dy_sum = np.sum(dy, axis=self.axis, keepdims=True)
        return dy - self._softmax * dy_sum

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
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
        params: list[np.ndarray] = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads: list[np.ndarray] = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return grads


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Токмаков Дмитрий Евгеньевич, ПМ-31"

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
