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


class MSELoss(Loss):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        n = self._x.size
        return 2 * (self._x - self._y) / n


class BCELoss(Loss):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        eps = 1e-15
        p = np.clip(x, eps, 1 - eps)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.mean(loss)

    def backward(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        n = self._x.shape[0]
        eps = 1e-15
        p = np.clip(self._x, eps, 1 - eps)
        dx = (p - self._y) / (p * (1 - p)) / n
        return dx


class NLLLoss(Loss):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        n = x.shape[0]
        loss = -np.mean(x[np.arange(n), y.astype(int)]) if y.ndim == 1 else -np.mean(np.sum(y * x, axis=-1))
        return loss

    def backward(self) -> np.ndarray:
        assert self._x is not None and self._y is not None
        n = self._x.shape[0]
        dx = np.zeros_like(self._x)
        if self._y.ndim == 1:
            dx[np.arange(n), self._y.astype(int)] = -1 / n
        else:
            dx = -self._y / n
        return dx


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._softmax: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        n = x.shape[0]

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)

        log_softmax = (x - x_max) - np.log(sum_exp)
        self._softmax = np.exp(log_softmax)

        loss = (
            -np.mean(log_softmax[np.arange(n), y.astype(int)])
            if y.ndim == 1
            else -np.mean(np.sum(y * log_softmax, axis=-1))
        )
        return loss

    def backward(self) -> np.ndarray:
        assert self._softmax is not None and self._y is not None and self._x is not None
        n = self._x.shape[0]
        dx = self._softmax.copy()

        if self._y.ndim == 1:
            dx[np.arange(n), self._y.astype(int)] -= 1
        else:
            dx -= self._y

        return dx / n


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
    def create_model(*layers: Layer) -> Model:
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
        model: Layer,
        loss: Loss,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        n_epoch: int,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        n_samples = x.shape[0]
        for _ in range(n_epoch):
            if shuffle:
                indices = np.random.permutation(n_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)
                dloss = loss.backward()
                model.backward(dloss)

                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
