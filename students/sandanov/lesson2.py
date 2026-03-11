import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + np.dot(x, self.weights)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return np.mean((y_pred - y) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        n = len(y)
        grad_w = (2 / n) * np.dot(x.T, (y_pred - y))
        grad_b = (2 / n) * np.sum(y_pred - y)
        return grad_w, grad_b


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        y_pred = (p >= 0.5).astype(float)
        return float(np.mean(y_pred == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(x)
        n = len(y)
        grad_w = (1 / n) * np.dot(x.T, (p - y))
        grad_b = (1 / n) * np.sum(p - y)
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Санданов Чимит Сергеевич, ПМ-34"

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
