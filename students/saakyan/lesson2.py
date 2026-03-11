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
        ost = y - self.predict(x)
        return np.mean(ost**2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        ss_res = np.sum((y - self.predict(x)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = y - self.predict(x)
        n = x.shape[0]
        dw = (-2 / n) * x.T @ errors
        db = (-2 / n) * np.sum(errors)
        return dw, db


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        st = x @ self.weights + self.bias
        sigmoid = 1 / 1 + np.exp(-st)
        return sigmoid

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        pred = np.clip(self.predict(x), eps, 1 - eps)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x) >= 0.5
        return np.mean(pred == y)

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = self.predict(x) - y
        n = x.shape[0]
        dw = x.T @ errors * (1 / n)
        db = np.sum(errors) * (1 / n)
        return dw, db


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Саакян Айк Алексанович, ПМ-34"

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
