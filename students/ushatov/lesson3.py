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
        self.x: np.ndarray | None = None
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.x is not None, "LinearLayer: incorrect forward and backward order"
        self.grad_weights = dy.T @ self.x
        self.grad_bias = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return self.weights, self.bias

    @property
    def grad(self) -> Sequence[np.ndarray]:
        assert self.grad_weights is not None and self.grad_bias is not None, (
            "LinearLaye: backward must be called before accessing grad"
        )
        return self.grad_weights, self.grad_bias


class ReLULayer(Layer):
    def __init__(self):
        self.logic_massive: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.logic_massive = x > 0
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.logic_massive  # np.multiply(dy, self.logic_massive, out=dy)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.y is not None, "SigmoidLayer: incorrect forward and backward order"
        return dy * self.y * (1 - self.y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        y = (x - x_max) - np.log(sum_exp)
        self.y = y
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self.y is not None, "LogSoftmaxLayer: incorrect forward and backward order"
        softmax = np.exp(self.y)
        sum_dy = np.sum(dy, axis=-1, keepdims=True)
        return dy - softmax * sum_dy

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer):
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


class MSELoss(Loss):
    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        return np.array(np.mean(np.square(x - y)))

    def backward(self) -> np.ndarray:
        assert self.x is not None and self.y is not None, "MSELoss: incorrect forward and backward order"
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    """Binary Cross Entropy Loss"""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        return np.array(-np.mean(y * np.log(self.x) + (1 - y) * np.log(1 - self.x)))

    def backward(self) -> np.ndarray:
        assert self.x is not None and self.y is not None, "BCELoss: incorrect forward and backward order"
        znam = self.x * (1 - self.x)
        assert np.all(znam != 0), "BCELoss: zero elements"
        return ((self.x - self.y) / znam) / self.x.shape[0]


class NLLLoss(Loss):
    """Negative Log-Likelihood Loss"""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        cache = x[np.arange(y.shape[0]), y]
        return np.array(-np.mean(cache))

    def backward(self) -> np.ndarray:
        assert self.x is not None and self.y is not None, "NLLLoss: incorrect forward and backward order"
        grad = np.zeros_like(self.x)
        grad[np.arange(self.y.shape[0]), self.y] = -1.0 / self.y.shape[0]
        return grad


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        self.gradient = np.array([])
        self.loss = np.array(0.0)

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        shape = x.shape[0]
        cashe = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(cashe)
        log_softmax = cashe - np.log(np.sum(exp_x, axis=-1, keepdims=True))
        one_hot = np.zeros_like(x)
        one_hot[np.arange(shape), y] = 1.0
        self.loss = -np.sum(log_softmax * one_hot) / shape
        self.gradient = (np.exp(log_softmax) - one_hot) / shape

        return np.array(self.loss)

    def backward(self) -> np.ndarray:
        return self.gradient


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Ушатов Сергей Максимович, ПМ-31"

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
        n = x.shape[0]
        batch = n if (batch_size is None or batch_size >= n) else batch_size

        for _ in range(n_epoch):
            for start in range(0, n, batch):
                end = min(start + batch, n)
                x_batch = x[start:end]
                y_batch = y[start:end]

                predict = model.forward(x_batch)
                loss.forward(predict, y_batch)

                dy = loss.backward()
                model.backward(dy)

                # Обновление параметров
                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
