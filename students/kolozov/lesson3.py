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

        self._weights_grad = np.zeros_like(self.weights)
        self._bias_grad = np.zeros_like(self.bias)

        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x

        output = x @ self.weights.T + self.bias

        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:

        if self._input_cache is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        x = self._input_cache

        self._weights_grad = dy.T @ x

        self._bias_grad = np.sum(dy, axis=0)

        dx = dy @ self.weights

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return [self._weights_grad, self._bias_grad]


class ReLULayer(Layer):
    def __init__(self) -> None:
        self._input_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x > 0

        output = np.maximum(0, x)

        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        dx = dy * self._input_cache

        return dx

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
            raise RuntimeError("forward() должен быть вызван перед backward()")

        sigmoid_output = self._output_cache

        grad_sigmoid = sigmoid_output * (1 - sigmoid_output)

        dx = dy * grad_sigmoid

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class LogSoftmaxLayer(Layer):
    def __init__(self, axis: int = -1) -> None:
        self._axis = axis
        self._output_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        c = np.max(x, axis=self._axis, keepdims=True)

        exp_shifted = np.exp(x - c)

        sum_exp = np.sum(exp_shifted, axis=self._axis, keepdims=True)

        output = x - c - np.log(sum_exp)

        self._output_cache = output

        return output

    def backward(self, dy: np.ndarray) -> np.ndarray:

        if self._output_cache is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        logsoftmax_output = self._output_cache

        softmax_output = np.exp(logsoftmax_output)

        dy_sum = np.sum(dy, axis=self._axis, keepdims=True)

        dx = dy - softmax_output * dy_sum

        return dx

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model(Layer):
    def __init__(self, *layers: Layer) -> None:

        self._layers = layers

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


class MSELoss(Loss):
    def __init__(self) -> None:
        self._grad: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x - y
        loss = np.mean(diff**2)

        n = x.size
        self._grad = 2 * diff / n

        return loss

    def backward(self) -> np.ndarray:
        if self._grad is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        return self._grad


class BCELoss:
    def __init__(self) -> None:
        self._grad: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        eps = 1e-13

        x_safe = x + eps

        loss = -np.mean(y * np.log(x_safe) + (1 - y) * np.log(1 - x_safe + eps))

        batch_size = x.shape[0]
        self._grad = (x_safe - y) / (x_safe * (1 - x_safe + eps) * batch_size)

        return loss

    def backward(self) -> np.ndarray:
        if self._grad is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        return self._grad


class NLLLoss:
    def __init__(self) -> None:
        self._grad: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        loss = -np.mean(x[np.arange(batch_size), y])

        self._grad = np.zeros_like(x)
        self._grad[np.arange(batch_size), y] = -1 / batch_size

        return loss

    def backward(self) -> np.ndarray:
        if self._grad is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")

        return self._grad


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._grad: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        c = np.max(x, axis=-1, keepdims=True)
        exp_shifted = np.exp(x - c)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)

        log_softmax = x - c - np.log(sum_exp)

        loss = -np.mean(log_softmax[np.arange(batch_size), y])

        softmax = np.exp(log_softmax)
        self._grad = softmax.copy()
        self._grad[np.arange(batch_size), y] -= 1
        self._grad /= batch_size

        return loss

    def backward(self) -> np.ndarray:
        if self._grad is None:
            raise RuntimeError("forward() должен быть вызван перед backward()")
        return self._grad


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Колосов Константин Николаевич, ПМ-33"

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
        n_batches = (n_samples + batch_size - 1) // batch_size

        for _ in range(n_epoch):
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                x_batch = x[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                predictions = model.forward(x_batch)
                loss.forward(predictions, y_batch)

                loss_grad = loss.backward()
                model.backward(loss_grad)

                params = model.parameters
                grads = model.grad

                for param, grad in zip(params, grads, strict=True):
                    param -= lr * grad
