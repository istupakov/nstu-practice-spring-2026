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
        y_prediction = self.predict(x)
        return float(np.mean((y - y_prediction) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x)
        P_2 = 1 - np.sum((y - y_prediction) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return float(P_2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        MSE = y_prediction - y
        n = len(y)
        grad_weights = (2 / n) * (x.T @ MSE)
        grad_bias = (2 / n) * np.sum(MSE)
        return grad_weights, grad_bias


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
        y_prediction = self.predict(x)
        eps = np.finfo(float).eps
        y_prediction = np.clip(y_prediction, eps, 1 - eps)
        return float(-np.sum(y * np.log(y_prediction) + (1 - y) * np.log(1 - y_prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_prediction = self.predict(x) >= 0.5
        n_correct = np.sum(y_prediction == y)
        total = len(y)
        return float(n_correct / total)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_prediction = self.predict(x)
        error = y_prediction - y
        n = len(y)
        grad_weights = (1 / n) * (x.T @ error)
        grad_bias = (1 / n) * np.sum(error)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Марченко Вячеслав Иванович, ПМ-33"

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
            grad_weight, grad_bias = model.grad(x, y)

            model.weights -= lr * grad_weight
            model.bias -= lr * grad_bias
