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
        return np.sum((y - self.predict(x)) ** 2) / (y.size)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        ssres: float = np.sum((y - self.predict(x)) ** 2)
        sstot: float = np.sum((y - np.mean(y)) ** 2)

        return 1 - ssres / sstot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n: float = y.size
        return -2 / n * (x.T @ (y - self.predict(x))), -2 / n * np.sum(y - self.predict(x))


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        return np.sum(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = np.where(self.predict(x) >= 0.5, 1, 0)

        return np.sum(y_pred == y) / y.size

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        return x.T @ (p - y), np.sum(p - y)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Придатченко Павел Павлович, ПМ-34"

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
            grad_w, grad_b = model.grad(x, y)
            model.weights -= lr * grad_w
            model.bias -= lr * grad_b
