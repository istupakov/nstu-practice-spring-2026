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

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        # Инициализируем пустым массивом вместо None, чтобы тайп-чекер не ругался
        self._input_cache = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.d_weights = dy.T @ self._input_cache
        self.d_bias = np.sum(dy, axis=0)
        dx = dy @ self.weights
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.d_weights, self.d_bias)


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._active_mask = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._active_mask = (x > 0).astype(x.dtype)
        return x * self._active_mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._active_mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self) -> None:
        self._activated_val = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._activated_val = 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))
        return self._activated_val

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._activated_val * (1.0 - self._activated_val)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self) -> None:
        self._softmax_probs = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        shifted_x = x - x_max
        exp_vals = np.exp(shifted_x)
        sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)

        self._softmax_probs = exp_vals / sum_exp
        return shifted_x - np.log(sum_exp)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - self._softmax_probs * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:
        self._sequence = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        current_x = x
        for layer in self._sequence:
            current_x = layer.forward(current_x)
        return current_x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        current_dy = dy
        for layer in reversed(self._sequence):
            current_dy = layer.backward(current_dy)
        return current_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params = []
        for layer in self._sequence:
            params.extend(layer.parameters)
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self._sequence:
            grads.extend(layer.grad)
        return tuple(grads)


class Exercise:
    @staticmethod
    def get_student() -> str:
        # ЗАМЕНИ НА СВОЕ ИМЯ
        return "Дегтярев Кирилл Романович, ПМ-35"

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
