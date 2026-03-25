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

        self._input_cache: np.ndarray | None = None
        self._grad_weights: np.ndarray | None = None
        self._grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise RuntimeError("LinearLayer: forward() must be called before backward()")

        self._grad_weights = dy.T @ self._input_cache
        self._grad_bias = np.sum(dy, axis=0)

        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        if self._grad_weights is None or self._grad_bias is None:
            raise RuntimeError("LinearLayer: backward() must be called before accessing grad")
        return [self._grad_weights, self._grad_bias]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._mask_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask_cache = x > 0
        return np.maximum(0, x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._mask_cache is None:
            raise RuntimeError("ReLULayer: forward() must be called before backward()")
        return dy * self._mask_cache

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = 1 / (1 + np.exp(-x))
        self._output_cache = output
        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._output_cache is None:
            raise RuntimeError("SigmoidLayer: forward() must be called before backward()")

        sigmoid_x = self._output_cache
        return dy * sigmoid_x * (1 - sigmoid_x)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self, axis: int = -1) -> None:
        self._axis = axis
        self._softmax_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=self._axis, keepdims=True)
        exp_shifted = np.exp(x - x_max)
        sum_exp = np.sum(exp_shifted, axis=self._axis, keepdims=True)

        self._softmax_cache = exp_shifted / sum_exp
        return (x - x_max) - np.log(sum_exp)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._softmax_cache is None:
            raise RuntimeError("LogSoftmaxLayer: forward() must be called before backward()")

        sum_dy = np.sum(dy, axis=self._axis, keepdims=True)
        return dy - self._softmax_cache * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
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
        return "Кудрявцев Павел Павлович, ПМ-35"

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
