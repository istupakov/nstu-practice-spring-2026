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
        pred = self.predict(x)
        mse = np.mean((pred - y) ** 2)
        return float(mse)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(1 - (np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        n = len(x)
        dw = -2 / n * (x.T @ (y - pred))
        db = -2 / n * np.sum(y - pred)
        return dw, db


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
        pred = self.predict(x)
        return float(np.mean(-y * np.log(pred) - (1 - y) * np.log(1 - pred)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return np.mean((pred >= 0.5).astype(int) == y)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x)
        n = len(x)
        dw = (1 / n) * (x.T @ (pred - y))
        db = (1 / n) * np.sum(pred - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Миллер Игорь Владиславович, ПМ-31"

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
        for _iteration in range(n_iter):
            dw, db = model.grad(x, y)
            model.weights -= lr * dw
            model.bias -= lr * db
