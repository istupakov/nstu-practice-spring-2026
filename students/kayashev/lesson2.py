import numpy as np


class LinearRegression:
    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weights) + self.bias

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.square(self.predict(x) - y)) / y.size

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        return 1 - np.sum((y - prediction) ** 2) / np.sum((y - np.average(y)) ** 2)

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.predict(x)
        return -2 / y.size * (y - prediction) @ x, -2 * np.mean(y - prediction)


class LogisticRegression:
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_features: int, rng: np.random.Generator) -> None:
        self.weights = rng.random(num_features)
        self.bias = np.array(0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        return 1 / (1 + (np.exp(-z)))

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        predict = self.predict(x)
        return -np.sum(y * np.log(predict) + (-y + 1) * np.log(-predict + 1))

    def metric(self, x: np.ndarray, y: np.ndarray) -> float:
        alpha = 0.5
        predict = self.predict(x)
        predictgrtthan = predict > alpha
        ygrtthan = y > alpha
        return np.sum(np.logical_not(np.logical_xor(predictgrtthan, ygrtthan).astype(int))) / y.size

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        predict = self.predict(x)
        return (y / predict - (1 - y) / (1 - predict)) @ x, np.sum(y / predict - (1 - y) / (1 - predict))


class Exercise:
    @staticmethod
    def get_student() -> str:
        return "Каяшев Валентин Константинович, ПМ-31"

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
            gradweight, gradbias = model.grad(x, y)
            model.weights -= lr * gradweight
            model.bias -= lr * gradbias
