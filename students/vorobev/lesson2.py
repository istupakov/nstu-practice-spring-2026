import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.square(y - self.predict(x))) / (y.size)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (np.sum(np.square(y - self.predict(x))) / np.sum(np.square(y - np.sum(y) / y.size)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        db = -2 * np.sum(y - self.predict(x)) / y.size
        dw = -2 * x.T @ (y - self.predict(x)) / y.size
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.bias + x @ self.weights)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.sum(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x))) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        doorstep = 0.5
        return np.sum((self.predict(x) >= doorstep).astype(int) == y) / y.size

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        db = np.sum(self.predict(x) - y) / y.size
        dw = (x.T @ (self.predict(x) - y)) / len(y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Воробьев Никита Александрович, ПМ-31"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng or np.random.default_rng())

    @staticmethod
    def fit(model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int) -> None:
        for _ in range(n_iter):
            dw, db = model.grad(x, y)
            model.weights -= lr * dw
            model.bias -= lr * db
