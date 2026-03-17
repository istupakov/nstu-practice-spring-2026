import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.bias + x @ self.weights

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        return float(np.mean((y - prediction) ** 2))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        sum1 = np.sum((y - prediction) ** 2)
        sum2 = np.sum((y - np.mean(y)) ** 2)
        return 1 - sum1 / sum2

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        grad_bias = -2 * np.mean(y - prediction)
        grad_weights = -2 * np.mean(x.T * (y - prediction), axis=1)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.bias + x @ self.weights
        return 1 / (1 + np.exp(-z))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return np.sum(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        accuracy = (prediction >= 0.5).astype(int)
        return float(np.mean(accuracy == y))

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        grad_bias = np.sum(prediction - y)
        grad_weights = x.T @ (prediction - y)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Пантеева Валентина Ивановна, ПМ-33"

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
            grad_weights, grad_bias = model.grad(x, y)

            model.weights -= lr * grad_weights
            model.bias -= lr * grad_bias
