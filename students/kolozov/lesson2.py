import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_part = x @ self.weights
        y_pred = linear_part + self.bias
        return y_pred

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        errors = y_pred - y
        squared_errors = errors**2
        return np.mean(squared_errors)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        y_pred = self.predict(x)
        error = y_pred - y  # знак противоположен теории, но компенсируется знаком ниже
        grad_weights = (2 / n) * (x.T @ error)
        grad_bias = (2 / n) * np.sum(error)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_part = x @ self.weights
        z = linear_part + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        return y_pred

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(x)
        p = np.clip(p, 1e-15, 1 - 1e-15)  # защита от log(0)
        term1 = y * np.log(p)
        term2 = (1 - y) * np.log(1 - p)
        return -np.sum(term1 + term2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        y_pred_class = (y_pred >= 0.5).astype(int)
        correct_pred = y_pred_class == y
        accuracy = np.mean(correct_pred)
        return accuracy

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(x)
        error = y_pred - y
        grad_weights = x.T @ error
        grad_bias = np.sum(error)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Колосов Константин Николаевич, ПМ-33"

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
        for _i in range(n_iter):
            grad_weights, grad_bias = model.grad(x, y)

            model.weights -= lr * grad_weights
            model.bias -= lr * grad_bias
