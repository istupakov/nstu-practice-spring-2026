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
        residuals = y - self.predict(x)
        return float(np.mean(residuals**2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        residuals = y - self.predict(x)
        total = y - np.mean(y)
        return float(1 - np.sum(residuals**2) / np.sum(total**2))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        residuals = self.predict(x) - y
        n = len(y)
        grad_w = 2 * (x.T @ residuals) / n
        grad_b = np.array(2 * np.mean(residuals))
        return grad_w, grad_b


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict(x), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = (self.predict(x) >= 0.5).astype(int)
        return float(np.mean(predictions == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        residuals = self.predict(x) - y
        n = len(y)
        grad_w = (x.T @ residuals) / n
        grad_b = np.array(np.mean(residuals))
        return grad_w, grad_b


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Наумов Дмитрий Сергеевич, ПМ-33"

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
            model.weights = model.weights - lr * grad_w
            model.bias = model.bias - lr * grad_b
