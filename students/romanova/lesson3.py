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
        self.W = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.b = rng.uniform(-k, k, out_features).astype(np.float32)

        self.W_grad: np.ndarray
        self.b_grad: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.W_grad = dy.T @ self.x
        self.b_grad = np.sum(dy, axis=0)
        return dy @ self.W

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.W, self.b)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.W_grad, self.b_grad)


class ReLULayer(Layer):
    def __init__(self):
        self.input_data: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_data = x
        return np.where(x > 0, x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        mask = (self.input_data > 0).astype(np.float32)
        return dy * mask

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.activation: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation = 1.0 / (1.0 + np.exp(-x))
        return self.activation

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.activation * (1 - self.activation)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.log_probs: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_centered = x - np.max(x, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x_centered), axis=-1, keepdims=True))
        self.log_probs = x_centered - log_sum_exp
        return self.log_probs

    def backward(self, dy: np.ndarray) -> np.ndarray:
        probs = np.exp(self.log_probs)
        return dy - probs * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
        self.network = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x
        for layer in self.network:
            current = layer.forward(current)
        return current

    def backward(self, dy: np.ndarray) -> np.ndarray:
        grad = dy
        for layer in reversed(self.network):
            grad = layer.backward(grad)
        return grad

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        result = []
        for layer in self.network:
            result.extend(layer.parameters)
        return tuple(result)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        result = []
        for layer in self.network:
            result.extend(layer.grad)
        return tuple(result)


class MSELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.array(np.mean((x - y) ** 2))

    def backward(self) -> np.ndarray:
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        eps = 1e-15
        x_clipped = np.clip(x, eps, 1 - eps)
        return np.array(-np.mean(y * np.log(x_clipped) + (1 - y) * np.log(1 - x_clipped)))

    def backward(self) -> np.ndarray:
        eps = 1e-15
        x_clipped = np.clip(self.x, eps, 1 - eps)
        return (x_clipped - self.y) / (x_clipped * (1 - x_clipped)) / self.x.shape[0]


class NLLLoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray
        self.batch_size: int

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]

        hot_y = np.zeros_like(x)
        hot_y[np.arange(self.batch_size), y] = 1

        return -np.sum(x * hot_y) / self.batch_size

    def backward(self) -> np.ndarray:
        hot_y = np.zeros_like(self.x)
        hot_y[np.arange(self.batch_size), self.y] = 1

        return -hot_y / self.batch_size


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.log_probs: np.ndarray
        self.y: np.ndarray
        self.batch_size: int

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.y = y
        self.batch_size = x.shape[0]

        x_centered = x - np.max(x, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x_centered), axis=-1, keepdims=True))
        self.log_probs = x_centered - log_sum_exp

        hot_y = np.zeros_like(x)
        hot_y[np.arange(self.batch_size), y] = 1

        return -np.sum(self.log_probs * hot_y) / self.batch_size

    def backward(self) -> np.ndarray:
        probs = np.exp(self.log_probs)
        hot_y = np.zeros_like(probs)
        hot_y[np.arange(self.batch_size), self.y] = 1

        return (probs - hot_y) / self.batch_size


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Романова Валерия Сергеевна, ПМ-34"

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
                    p -= lr * g
