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

        self._input: np.ndarray = np.empty(0, dtype=np.float32)
        self._weights_grad: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._bias_grad: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход: y = x * W^T + b"""
        self._input = x

        return np.dot(x, self.weights.T) + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:

        # Градиент по входу: dx = dy * W
        dx = np.dot(dy, self.weights)

        # Градиент по весам: dW = dy^T * x
        self._weights_grad = np.dot(dy.T, self._input)

        # Градиент по смещениям: db = сумма dy по батчу
        self._bias_grad = np.sum(dy, axis=0)

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self._weights_grad, self._bias_grad]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._mask: np.ndarray = np.empty(0, dtype=bool)  #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход: оставляем положительные значения, отрицательные обнуляем"""
        self._mask = x > 0
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Обратный проход: градиент проходит только через положительные значения"""
        return dy * self._mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._output: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход: вычисляем сигмоиду"""
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Обратный проход: dy * f(x) * (1 - f(x))"""
        return dy * self._output * (1 - self._output)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis  #
        self._softmax: np.ndarray = np.empty(0, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход: вычисляем log(softmax(x))"""
        x_max = np.max(x, axis=self.axis, keepdims=True)

        exp_x = np.exp(x - x_max)

        log_sum_exp = np.log(np.sum(exp_x, axis=self.axis, keepdims=True))

        output = x - x_max - log_sum_exp

        self._softmax = np.exp(output)

        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Обратный проход: градиент для log(softmax(x))
        Формула: dy - softmax * sum(dy)
        """
        sum_dy = np.sum(dy, axis=self.axis, keepdims=True)

        return dy - self._softmax * sum_dy

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


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Разин Игорь Дмитриевич, ПМ-33"

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
