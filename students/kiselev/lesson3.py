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


class Loss(Protocol):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def backward(self) -> np.ndarray: ...


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x: np.ndarray | None = None
        self._dw: np.ndarray | None = None
        self._db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        self._dw = dy.T @ self._x
        self._db = dy.sum(axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        if self._dw is None or self._db is None:
            return ()
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("forward must be called before backward")
        return dy * self._mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer:
    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("forward must be called before backward")
        return dy * self._out * (1.0 - self._out)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer:
    def __init__(self) -> None:
        self._log_softmax: np.ndarray | None = None

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
    def __init__(self, *layers: Layer) -> None:
        self._layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self._layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        params = []
        for layer in self._layers:
            params.extend(layer.parameters)
        return params

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self._layers:
            grads.extend(layer.grad)
        return grads


_EPS = 1e-9


class MSELoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x, self._y = x, y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        return 2 * (self._x - self._y) / self._x.size


class BCELoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x, self._y = x, y
        return -np.mean(y * np.log(x + _EPS) + (1 - y) * np.log(1 - x + _EPS))

    def backward(self) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        return (self._x - self._y) / (self._x * (1 - self._x) + _EPS) / self._x.shape[0]


class NLLLoss:
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x, self._y = x, y
        n = x.shape[0]
        return -np.mean(x[np.arange(n), y])

    def backward(self) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        n = self._x.shape[0]
        grad = np.zeros_like(self._x)
        grad[np.arange(n), self._y] = -1.0 / n
        return grad


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._log_probs: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._y = y
        n = x.shape[0]
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        self._log_probs = x_shifted - np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
        return -np.mean(self._log_probs[np.arange(n), y])

    def backward(self) -> np.ndarray:
        if self._log_probs is None:
            raise RuntimeError("forward must be called before backward")
        n = self._log_probs.shape[0]
        grad = np.exp(self._log_probs)
        grad[np.arange(n), self._y] -= 1.0
        return grad / n


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Киселев Эдуард Владиславович, ПМ-33"

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

    @staticmethod
    def create_mse_loss() -> Loss:
        return MSELoss()

    @staticmethod
    def create_bce_loss() -> Loss:
        return BCELoss()

    @staticmethod
    def create_nll_loss() -> Loss:
        return NLLLoss()

    @staticmethod
    def create_cross_entropy_loss() -> Loss:
        return CrossEntropyLoss()

    @staticmethod
    def train_model(
        model: Layer, loss: Loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int
    ) -> None:
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for _ in range(n_epoch):
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)

                d_loss = loss.backward()
                model.backward(d_loss)

                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
