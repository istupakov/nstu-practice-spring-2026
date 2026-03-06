import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)  # смещение

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.square(y - self.predict(x)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - self.loss(x, y) / np.var(y)  # / дисперсию

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        dw = (-2 / x.shape[0]) * np.dot(x.T, (y - p))
        db = -2 * np.mean(y - p)
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return 1 / np.exp(-z)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -np.sum(y * np.log(self.predict(x)) + (1 - y) * np.log(1 - self.predict(x)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((self.predict(x) >= 0.5) == y)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:

        p = self.predict(x)
        f = p / (p + 1)
        db = f * (1 - f)
        dw = x * db
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Кузьмин Александр Андреевич, ПМ-35"

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
