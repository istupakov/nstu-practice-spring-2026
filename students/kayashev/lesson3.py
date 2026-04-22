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
        assert rng is not None
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self.dw: np.ndarray
        self.db: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self.x
        self.dw = dy.T @ x
        self.db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self.dw, self.db)


class ReLULayer(Layer):
    def __init__(self):
        self.x: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * (self.x > 0).astype(np.float32)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class SigmoidLayer(Layer):
    def __init__(self):
        self.y: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.y * (1 - self.y)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self):
        self.y: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        newx = x - np.max(x, axis=-1, keepdims=True)
        self.y = newx - np.log(np.sum(np.exp(newx), axis=-1, keepdims=True))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy - (np.exp(self.y) * np.sum(dy, axis=-1, keepdims=True))

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class MSELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return np.mean((x - y) ** 2)

    def backward(self) -> np.ndarray:
        return 2 * (self.x - self.y) / self.x.size


class BCELoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return -np.mean(y * np.log(x) + (-y + 1) * np.log(-x + 1))

    def backward(self) -> np.ndarray:
        return ((self.x - self.y) / (self.x * (1 - self.x))) / self.x.shape[0]


class NLLLoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray
        self.hot: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        self.hot = np.zeros_like(x)
        self.hot[np.arange(self.x.shape[0]), y] = 1
        return -np.sum(x * self.hot) / self.x.shape[0]

    def backward(self) -> np.ndarray:
        return -self.hot / self.x.shape[0]


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.x: np.ndarray
        self.y: np.ndarray
        self.log_softmax: np.ndarray
        self.hot: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        if y.ndim == 1:
            self.hot = np.zeros_like(x)
            self.hot[np.arange(x.shape[0]), y] = 1
            self.y = y
        else:
            self.hot = y
            self.y = np.argmax(y, axis=1)
        x_max = np.max(x, axis=-1, keepdims=True)
        shifted_x = x - x_max
        sum_exp = np.sum(np.exp(shifted_x), axis=-1, keepdims=True)
        self.log_softmax = shifted_x - np.log(sum_exp)
        return -np.sum(self.log_softmax * self.hot) / self.x.shape[0]

    def backward(self) -> np.ndarray:
        softmax = np.exp(self.log_softmax)
        return (softmax - self.hot) / self.x.shape[0]


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
        return tuple(params)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        grads = []
        for layer in self.layers:
            grads.extend(layer.grad)
        return tuple(grads)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Каяшев Валентин Константинович, ПМ-31"

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
    def create_model(*layers: Layer) -> Layer:
        return Model(*layers)

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
        dims = x.shape[0]

        for _ in range(n_epoch):
            if shuffle:
                indices = np.random.permutation(dims)
                newx = x[indices]
                newy = y[indices]
            else:
                newx = x
                newy = y
            for start in range(0, dims, batch_size):
                end_idx = min(start + batch_size, dims)
                batchx = newx[start:end_idx]
                batchy = newy[start:end_idx]
                predictions = model.forward(batchx)
                # print("Test ",predictions,"\n")
                loss.forward(predictions, batchy)
                lossback = loss.backward()
                model.backward(lossback)
                for param, grad in zip(model.parameters, model.grad, strict=True):
                    param -= lr * grad
