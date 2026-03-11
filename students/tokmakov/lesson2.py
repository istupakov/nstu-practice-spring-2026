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
        n = len(x)
        return (np.sum(np.square(y - self.predict(x)))) / n

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1 - np.sum(np.square(y - self.predict(x))) / np.sum(np.square(y - np.mean(y)))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(x)
        predictions = self.predict(x)
        errors = y - predictions
        grad_w = (-2 / n) * x.T @ errors
        grad_b = (-2 / n) * np.sum(errors)
        grad_b = np.array(grad_b)

        return grad_w, grad_b


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
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.mean(losses))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        predictions = p >= 0.5
        return float(np.mean(predictions == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = x.shape[0]
        p = self.predict(x)
        grad_w = (1 / n) * x.T @ (p - y)
        grad_b = (1 / n) * np.sum(p - y)
        return grad_w, np.array(grad_b)


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Токмаков Дмитрий Евгеньевич, ПМ-31"

    @staticmethod
    def get_topic() -> str:
        return "Lesson 2"

    @staticmethod
    def create_linear_model(num_features: int, rng: np.random.Generator | None = None) -> LinearRegression:
        return LinearRegression(num_features, rng=rng or np.random.default_rng())

    @staticmethod
    def create_logistic_model(num_features: int, rng: np.random.Generator | None = None) -> LogisticRegression:
        return LogisticRegression(num_features, rng=rng or np.random.default_rng())

    @staticmethod
    def fit(
        model: LinearRegression | LogisticRegression, x: np.ndarray, y: np.ndarray, lr: float, n_iter: int
    ) -> None: ...
