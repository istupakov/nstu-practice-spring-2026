from collections.abc import Sequence

import numpy as np

# ─── Activation / Layer helpers ──────────────────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


# ─── Layers ──────────────────────────────────────────────────────────────────


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        k = np.sqrt(1 / in_features)
        self.weights = rng.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.bias = rng.uniform(-k, k, out_features).astype(np.float32)
        self._x: np.ndarray | None = None
        self._dw: np.ndarray = np.zeros_like(self.weights)
        self._db: np.ndarray = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.weights.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self._dw = dy.T @ self._x
        self._db = np.sum(dy, axis=0)
        return dy @ self.weights

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return (self.weights, self.bias)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return (self._dw, self._db)


class ReLULayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.maximum(x, 0)
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
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
        self._y = _sigmoid(x)
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dy * self._y * (1.0 - self._y)

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
        self._y = _log_softmax(x)
        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dy - np.exp(self._y) * np.sum(dy, axis=-1, keepdims=True)

    @property
    def parameters(self) -> Sequence[np.ndarray]:
        return ()

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return ()


class Model:
    def __init__(self, *layers) -> None:
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
        return tuple(p for layer in self._layers for p in layer.parameters)

    @property
    def grad(self) -> Sequence[np.ndarray]:
        return tuple(g for layer in self._layers for g in layer.grad)


# ─── Loss functions ───────────────────────────────────────────────────────────


class MSELoss:
    _x: np.ndarray
    _y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.asarray(np.mean((x - y) ** 2), dtype=np.float32)

    def backward(self) -> np.ndarray:
        return np.asarray(2.0 * (self._x - self._y) / self._x.size, dtype=np.float32)


class BCELoss:
    _x: np.ndarray
    _y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._x = x
        self._y = y
        return np.asarray(np.mean(-(y * np.log(x) + (1 - y) * np.log(1 - x))))

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        return (self._x - self._y) / (self._x * (1 - self._x)) / batch_size


class NLLLoss:
    _x: np.ndarray
    _y: np.ndarray
    _hot_y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # x: log-probabilities (batch, C), y: class indices (batch,)
        self._x = x
        self._y = y
        batch_size = x.shape[0]
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1
        self._hot_y = hot_y
        return (-np.sum(x * hot_y) / batch_size).astype(np.float32)

    def backward(self) -> np.ndarray:
        batch_size = self._x.shape[0]
        return (-self._hot_y / batch_size).astype(np.float32)


class CrossEntropyLoss:
    _logprobs: np.ndarray
    _y: np.ndarray
    _hot_y: np.ndarray

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # x: raw logits (batch, C), y: class indices (batch,)
        self._logprobs = _log_softmax(x)
        self._y = y
        batch_size = x.shape[0]
        hot_y = np.zeros_like(x)
        hot_y[np.arange(batch_size), y] = 1
        self._hot_y = hot_y
        return (-np.sum(self._logprobs * hot_y) / batch_size).astype(np.float32)

    def backward(self) -> np.ndarray:
        batch_size = self._logprobs.shape[0]
        return ((np.exp(self._logprobs) - self._hot_y) / batch_size).astype(np.float32)


# ─── Exercise ─────────────────────────────────────────────────────────────────


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Батодалаев Арсалан Дабаевич, ПМ-33"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 3"

    @staticmethod
    def create_linear_layer(in_features: int, out_features: int, rng: np.random.Generator | None = None) -> LinearLayer:
        return LinearLayer(in_features, out_features, rng)

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
    def create_model(*layers) -> Model:
        return Model(*layers)

    @staticmethod
    def create_mse_loss() -> MSELoss:
        return MSELoss()

    @staticmethod
    def create_bce_loss() -> BCELoss:
        return BCELoss()

    @staticmethod
    def create_nll_loss() -> NLLLoss:
        return NLLLoss()

    @staticmethod
    def create_cross_entropy_loss() -> CrossEntropyLoss:
        return CrossEntropyLoss()

    @staticmethod
    def train_model(model: Model, loss, x: np.ndarray, y: np.ndarray, lr: float, n_epoch: int, batch_size: int) -> None:
        idx = np.arange(batch_size, x.shape[0], batch_size)
        for _ in range(n_epoch):
            for x_batch, y_batch in zip(np.split(x, idx, axis=0), np.split(y, idx, axis=0), strict=True):
                loss.forward(model.forward(x_batch), y_batch)
                model.backward(loss.backward())
                for p, g in zip(model.parameters, model.grad, strict=True):
                    p += -lr * g