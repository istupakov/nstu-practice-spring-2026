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

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x

        return np.dot(x, self.weights.T) + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.grad_weights = np.dot(dy.T, self._x)
        self.grad_bias = np.sum(dy, axis=0)

        return np.dot(dy, self.weights)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.grad_weights, self.grad_bias)


class ReLULayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.positive = x > 0

        return np.where(self.positive, x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:

        return dy * self.positive

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1.0 / (1.0 + np.exp(-x))

        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.y * (1.0 - self.y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        c = np.max(x, axis=-1, keepdims=True)
        self.x_res = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
        return self.x_res

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - np.exp(self.x_res) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = [x]
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters)
        return parameters

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grad = []
        for layer in self.layers:
            grad.extend(layer.grad)
        return grad


class MSELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.mean(np.square(x - y))

    def backward(self) -> np.ndarray:
        n = self.x.size
        return 2 * (self.x - self.y) / n


class BCELoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return -np.mean(self.y * np.log(self.x) + (1 - self.y) * np.log(1 - self.x))

    def backward(self) -> np.ndarray:
        n = self.x.shape[0]
        return -(self.y / self.x - (1 - self.y) / (1 - self.x)) / n


class NLLLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        self.grad = np.zeros_like(x)
        self.grad[np.arange(n), y] = -1 / n

        return -np.mean(x[np.arange(n), y])

    def backward(self) -> np.ndarray:
        return self.grad


class CrossEntropyLoss(Loss):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]

        c = np.max(x, axis=-1, keepdims=True)
        log_soft_max = x - c - np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))

        self.probs = np.exp(log_soft_max)
        self.y = y

        return -np.mean(log_soft_max[np.arange(n), y])

    def backward(self) -> np.ndarray:
        n = self.probs.shape[0]
        grad = self.probs.copy()

        grad[np.arange(n), self.y] -= 1.0

        return grad / n


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Санданов Чимит Сергеевич, ПМ-34"

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

        x_train = x[indices]
        y_train = y[indices]

        for _ in range(n_epoch):
            for i in range(0, n_samples, batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)

                d_loss = loss.backward()
                model.backward(d_loss)

                params = model.parameters
                grads = model.grad

                for param, grad in zip(params, grads, strict=True):
                    param -= lr * grad
