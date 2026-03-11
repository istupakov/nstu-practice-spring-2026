import numpy as np


class LinearRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((y - self.predict(x)) ** 2)

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        up_sum = np.sum((y - self.predict(x)) ** 2)
        lower_sum = np.sum((y - np.mean(y)) ** 2)
        return 1 - up_sum / lower_sum

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        grad_weights = 2 / len(y) * np.matmul(x.T, (self.predict(x) - y))
        grad_bias = 2 / len(y) * np.sum(self.predict(x) - y)
        return grad_weights, grad_bias


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(np.matmul(x, (self.weights) + self.bias))))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        p_i = np.clip(self.predict(x), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p_i) + (1 - y) * np.log(1 - p_i))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((self.predict(x) >= 0.5).astype(int) == y)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        grad_weights = 1 / len(y) * np.matmul(x.T, (self.predict(x) - y))
        grad_bias = 1 / len(y) * np.sum(self.predict(x) - y)
        return grad_weights, grad_bias


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Старицын Марк Вадимович, ПМ-35"

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
