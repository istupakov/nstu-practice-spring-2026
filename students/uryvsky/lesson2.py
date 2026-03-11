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
        return np.mean(np.square(y - self.predict(x)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - (self.loss(x, y) / np.var(y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        dw = -2.0 * x.T @ (y - self.predict(x)) / x.shape[0]
        db = -2.0 * np.sum(y - self.predict(x)) / x.shape[0]
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
        eps = 1e-15
        p = np.clip(self.predict(x), eps, 1 - eps)
        return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        solution_predict = np.mean((self.predict(x) >= 0.5).astype(int) == y)
        return solution_predict

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        dw = (x.T @ (self.predict(x) - y)) / len(y)
        db = np.mean(self.predict(x) - y)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Урывский Александр Александрович, ПМ-31"

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
