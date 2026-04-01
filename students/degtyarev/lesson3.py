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

        self._d_weights = np.zeros_like(self.weights)
        self._d_bias = np.zeros_like(self.bias)
        self._input_cache = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self._d_weights = dy.T @ self._input_cache
        self._d_bias = np.sum(dy, axis=0)
        dx = dy @ self.weights
        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self._d_weights, self._d_bias)


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


class MSELoss(Loss):
    def __init__(self) -> None:
        self._x = np.array([])
        self._y = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.array(np.mean((x - y) ** 2), dtype=np.float32)

    def backward(self) -> np.ndarray:
        return 2.0 * (self._x - self._y) / self._x.size


class BCELoss(Loss):
    def __init__(self) -> None:
        self._x = np.array([])
        self._y = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        loss = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
        return np.array(loss)

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        return (self._x - self._y) / (self._x * (1 - self._x)) / batch_size


class NLLLoss(Loss):
    def __init__(self) -> None:
        self._dx = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1
        self._dx = -hot_y / batch_size
        loss = -np.sum(x * hot_y) / batch_size
        return np.array(loss, dtype=np.float32)

    def backward(self) -> np.ndarray:
        return self._dx


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        self._dx = np.array([])
        self._loss = np.array(0.0)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        shifted = x - np.max(x, axis=-1, keepdims=True)
        logprobs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1

        self._loss = -np.sum(logprobs * hot_y) / batch_size
        self._dx = (np.exp(logprobs) - hot_y) / batch_size
        return np.array(self._loss)

    def backward(self) -> np.ndarray:
        return self._dx


class Exercise:
    @staticmethod
    def get_student() -> str:
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
        idx = np.arange(batch_size, x.shape[0], batch_size)

        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())

                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g
