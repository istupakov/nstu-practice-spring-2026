from collections.abc import Sequence
from typing import Any

import numpy as np


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        k = float(np.sqrt(1.0 / in_features))
        self.w = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.b = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x: np.ndarray | None = None
        self._dw: np.ndarray | None = None
        self._db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.w.T + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")

        x = self._x
        self._dw = dy.T @ x
        self._db = np.sum(dy, axis=0)
        dx = dy @ self.w
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.w, self.b)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        if self._dw is None or self._db is None:
            return ()
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.maximum(x, 0)
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        return dy * np.sign(self._y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = 1 / (1 + np.exp(-x))
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        y = self._y
        return dy * y * (1 - y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        y = x_shift - np.log(np.sum(np.exp(x_shift), axis=-1, keepdims=True))
        self._y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("forward must be called before backward")
        y = self._y
        return dy - (np.exp(y) * np.sum(dy, axis=-1, keepdims=True))

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, layers: tuple[Any, ...]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        grad = dy
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params: list[np.ndarray] = []
        for layer in self.layers:
            params.extend(list(layer.parameters))
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads: list[np.ndarray] = []
        for layer in self.layers:
            grads.extend(list(layer.grad))
        return tuple(grads)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Мелиди Мирон Евстафьевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> LinearLayer:
        return LinearLayer(in_features, out_features, rng or np.random.default_rng())

    @staticmethod
    def create_relu_layer() -> ReLULayer:
        return ReLULayer()

    @staticmethod
    def create_sigmoid_layer() -> SigmoidLayer:
        return SigmoidLayer()

    @staticmethod
    def create_logsoftmax_layer() -> LogSoftmaxLayer:
        return LogSoftmaxLayer()

    @staticmethod
    def create_model(*layers: object) -> Model:
        return Model(tuple(layers))
